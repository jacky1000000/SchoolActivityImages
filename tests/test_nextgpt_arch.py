import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from models.nextgpt_arch import NextGPTImageGenModel


class NextGPTImageGenModelTests(unittest.TestCase):
    def setUp(self):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        with patch(
            "models.nextgpt_arch.AutoModelForCausalLM.from_pretrained",
            return_value=LlamaForCausalLM(config),
        ):
            self.model = NextGPTImageGenModel("unused", freeze_llm=False)
        self.model.llm = get_peft_model(
            self.model.llm,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=2,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        self.input_ids = torch.tensor([[1, 2, 3, 4]])

    def test_labels_produce_loss_and_lora_gradients(self):
        output = self.model(input_ids=self.input_ids, labels=self.input_ids)

        self.assertIsNotNone(output["loss"])
        self.assertTrue(torch.isfinite(output["loss"]).item())
        output["loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                for name, parameter in self.model.llm.named_parameters()
                if "lora_" in name
            )
        )
        self.assertEqual(output["image_features"].shape, (1, 4, 768))

    def test_without_labels_returns_no_loss(self):
        output = self.model(input_ids=self.input_ids)

        self.assertIsNone(output["loss"])
        self.assertEqual(output["logits"].shape, (1, 4, 32))

    def test_bfloat16_model_trains_with_autocast(self):
        self.model.llm.to(dtype=torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = self.model(input_ids=self.input_ids, labels=self.input_ids)

        self.assertIsNotNone(output["loss"])
        self.assertTrue(torch.isfinite(output["loss"]).item())
        output["loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                for name, parameter in self.model.llm.named_parameters()
                if "lora_" in name
            )
        )

    def test_model_dtype_is_passed_to_loader(self):
        self.model.llm.to(dtype=torch.bfloat16)
        with patch(
            "models.nextgpt_arch.AutoModelForCausalLM.from_pretrained",
            return_value=self.model.llm,
        ) as loader:
            model = NextGPTImageGenModel("unused", torch_dtype=torch.bfloat16)

        loader.assert_called_once_with("unused", torch_dtype=torch.bfloat16)
        output = model(input_ids=self.input_ids)
        self.assertEqual(output["image_features"].dtype, torch.bfloat16)

    def test_vocab_size_resize_allows_new_token_ids(self):
        base_model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        with patch(
            "models.nextgpt_arch.AutoModelForCausalLM.from_pretrained",
            return_value=base_model,
        ):
            model = NextGPTImageGenModel("unused", freeze_llm=False, vocab_size=33)

        output = model(input_ids=torch.tensor([[1, 2, 32]]))
        self.assertEqual(output["logits"].shape[-1], 33)


if __name__ == "__main__":
    unittest.main()
