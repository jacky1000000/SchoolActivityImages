import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data

class CampusRetrievalDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]["image_path"], self.pairs[idx]["text"]

def collate_fn(batch, device):
    img_paths = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    vision_tensors = imagebind_data.load_and_transform_vision_data(img_paths, device)
    text_tensors = imagebind_data.load_and_transform_text(texts, device)
    return vision_tensors, text_tensors

def symmetric_infonce_loss(vision_embeds, text_embeds, temperature=0.07):
    vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    logits = torch.matmul(vision_embeds, text_embeds.T) / temperature
    targets = torch.arange(vision_embeds.shape[0], device=vision_embeds.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2.0

def freeze_backbone_except_top(model):
    for param in model.parameters():
        param.requires_grad = False

    # 僅解凍輸出投影頭與最頂層 Transformer Block
    trainable_keywords = ["modality_heads", "blocks.31", "blocks.23", "norm"]
    trainable_params = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True
            trainable_params += param.numel()

    print(f"🔓 已解凍參數量: {trainable_params / 1e6:.2f} M")

def train_imagebind():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "../../my_checkpoints/imagebind_huge.pth"
    train_json = "../../data/imagebind_train.json"
    output_dir = "../../ckpt_updated/imagebind_campus_tuned"
    os.makedirs(output_dir, exist_ok=True)

    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    freeze_backbone_except_top(model)

    dataset = CampusRetrievalDataset(train_json)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, device)
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    num_epochs = 10

    print(f"🚀 開始微調 ImageBind (樣本數: {len(dataset)} | Epochs: {num_epochs})...")
    for epoch in range(num_epochs):
        print(f"🔄 開始 Epoch {epoch+1}/{num_epochs}...")
        model.train()
        total_loss = 0.0

        for vision_inputs, text_inputs in dataloader:
            optimizer.zero_grad()
            outputs = model({
                ModalityType.VISION: vision_inputs,
                ModalityType.TEXT: text_inputs
            })
            loss = symmetric_infonce_loss(outputs[ModalityType.VISION], outputs[ModalityType.TEXT])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | 平均對比 Loss: {total_loss / len(dataloader):.4f}")

    save_path = os.path.join(output_dir, "imagebind_campus_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"🎉 微調完成！權重已儲存至: {save_path}")

if __name__ == "__main__":
    train_imagebind()
