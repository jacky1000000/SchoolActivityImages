import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class NextGPTImageGenModel(nn.Module):
    def __init__(self, vicuna_path, freeze_llm=True, torch_dtype=None):
        super().__init__()
        # 1. 載入大語言模型核心 (Vicuna 7B)
        self.llm = AutoModelForCausalLM.from_pretrained(vicuna_path, torch_dtype=torch_dtype)
        # if vocab_size is not None and vocab_size != self.llm.config.vocab_size:
        #     self.llm.resize_token_embeddings(vocab_size)
        
        # 2. 圖像生成輸出連接層 (Output Projection Layer)
        # 將 Vicuna 的 4096 維隱藏向量映射到 Stable Diffusion 的 768 維 Text Embeddings 空間
        self.to_image_diffusion = nn.Linear(self.llm.config.hidden_size, 768, dtype=torch_dtype)
        
        # 根據下游任務需求決定是否凍結語言模型主幹
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, labels=None):
        # 執行 LLM 前向傳播
        outputs = self.llm(input_ids=input_ids, labels=labels, output_hidden_states=True)
        
        # 提取 LLM 最後一層的隱藏狀態，維度: [Batch, Sequence_Length, 4096]
        last_hidden_state = outputs.hidden_states[-1]
        
        # 將特徵映射到圖像擴散模型的特徵空間，維度: [Batch, Sequence_Length, 768]
        image_features = self.to_image_diffusion(last_hidden_state)
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "image_features": image_features
        }
