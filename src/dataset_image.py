import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CampusImageDataset(Dataset):
    def __init__(self, json_path, tokenizer_path, max_length=256):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # 載入與 Vicuna 配對的分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item["conversations"]
        
        # 將 User 的開放式提問與 Assistant 包含 <IMG0> 的回答組裝成訓練文本
        # full_text = f"USER: {conversation[0]['value']} ASSISTANT: {conversation[1]['value']}{self.tokenizer.eos_token}"
        full_text = f"USER: {conversation[0]['value']} ASSISTANT: {conversation[1]['value']}"
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
