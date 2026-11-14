# def main() -> None:
#     print("Hello from schoolactivityimages!")
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

from dataset_image import CampusImageDataset
from models.nextgpt_arch import NextGPTImageGenModel

# 訓練校園設施生成模型的函式
def train_facility_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Model dtype: {model_dtype}")
    os.makedirs("../ckpt_updated/nextgpt_facility_output", exist_ok=True)
    
    vicuna_path = "../my_checkpoints/vicuna-7b-v1.5"
    
    # 1. 初始化模型 (不凍結 LLM 主幹，因為稍後要外掛 LoRA)
    model = NextGPTImageGenModel(
        vicuna_path=vicuna_path, freeze_llm=False, torch_dtype=model_dtype
    )
    
    # 2. 注入 LoRA 配置，專注訓練注意力投影層
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    model.llm = get_peft_model(model.llm, peft_config)
    # 啟用梯度檢查點，大幅降低顯存佔用，避免系統拖慢速度
    model.llm.enable_input_require_grads()
    model.llm.gradient_checkpointing_enable()
    model.to(device)
    
    # 3. 準備開放式設施圖像數據集
    train_dataset = CampusImageDataset(
        json_path="../data/image_gen_train.json",
        tokenizer_path=vicuna_path,
        max_length=128
    )
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    print(f"訓練數據集大小: {len(train_dataset)}")
    # print('Show the content inside dataset: ')
    # for i in range(len(train_dataset)):
    #     print(train_dataset[i])
    
    # 4. 優化器設定
    optimizer = AdamW(model.parameters(), lr=2e-4)
    num_epochs = 25
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # 5. 訓練循環
    print("🚀 開始進行開放式校園設施生成微調...")
    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        print(f"開始 Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.autocast(device_type=device, dtype=model_dtype, enabled=use_bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
            print(f'outputs returned for epoch {epoch+1}')
            loss = outputs["loss"]

            # 檢查是否出現 NaN 異常
            if torch.isnan(loss):
                print("⚠️ 檢測到 NaN Loss，跳過此批次")
                continue
            
            loss.backward()
            # 關鍵防護：梯度裁剪，防止權重崩壞變成亂碼
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} | 平均訓練 Loss: {total_loss/len(train_loader):.4f} | {elapsed:.1f}s")

        # 只要 Loss 已經降到 0.6 以下，每輪自動覆蓋儲存最新權重，隨時可停下來測試
        if total_loss/len(train_loader) < 0.6:
            torch.save(model.to_image_diffusion.state_dict(), "../ckpt_updated/nextgpt_facility_output/image_projection.pth")
            model.llm.save_pretrained("../ckpt_updated/nextgpt_facility_output/vicuna_lora")
            print(f"💾 已自動備份第 {epoch+1} 輪有效權重！")
        
    # 6. 儲存微調後增量參數
    torch.save(model.to_image_diffusion.state_dict(), "../ckpt_updated/nextgpt_facility_output/image_projection.pth")
    model.llm.save_pretrained("../ckpt_updated/nextgpt_facility_output/vicuna_lora")
    print("🎉 微調成功結束！權重已儲存至 ../ckpt_updated/nextgpt_facility_output/")

def main() -> None:
    train_facility_model()

if __name__ == "__main__":
    main()