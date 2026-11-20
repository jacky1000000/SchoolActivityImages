import os
import json
import shutil
import torch
import torch.nn.functional as F
from collections import Counter
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data

def extract_all_photo_features(model, test_paths, device, batch_size=32):
    """一次性提取所有測試集照片的特徵矩陣，避免重複計算"""
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(test_paths), batch_size):
            batch = test_paths[i:i + batch_size]
            inputs = {ModalityType.VISION: imagebind_data.load_and_transform_vision_data(batch, device)}
            v_embeds = F.normalize(model(inputs)[ModalityType.VISION], p=2, dim=-1)
            all_embeds.append(v_embeds)
    return torch.cat(all_embeds, dim=0)

def query_model_candidates(ckpt_path, model_label, test_paths, pooling_tasks, top_k, device):
    """載入單一模型並為所有測試題目撈出候選檔名"""
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 找不到權重 '{ckpt_path}'，跳過 [{model_label}]。")
        return {task["query_id"]: [] for task in pooling_tasks}

    print(f"⏳ 正在載入 [{model_label}] 並提取測試集特徵...")
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)

    photo_matrix = extract_all_photo_features(model, test_paths, device)
    all_filenames = [os.path.basename(p) for p in test_paths]
    task_results = {}

    for task in pooling_tasks:
        q_id = task["query_id"]
        found_files = []
        for sq in task["pooling_prompts"]:
            with torch.no_grad():
                text_inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text([sq], device)}
                text_embed = F.normalize(model(text_inputs)[ModalityType.TEXT], p=2, dim=-1)

            similarities = torch.matmul(photo_matrix, text_embed.T).squeeze(-1)
            _, top_indices = torch.topk(similarities, k=min(top_k, len(test_paths)))
            for idx in top_indices.tolist():
                found_files.append(all_filenames[idx])
        task_results[q_id] = found_files

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return task_results

def build_fair_eval_pool(
    pooling_tasks,
    test_photo_dir="../../data/test_photos",
    base_ckpt="../../my_checkpoints/imagebind_huge.pth",
    tuned_ckpt="../../ckpt_updated/imagebind_campus_tuned/imagebind_campus_finetuned.pth",
    preview_root="../../output/fair_pooling_preview",
    draft_json_path="../../data/eval_queries_draft_test_data.json",
    top_k_per_query=10
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    test_paths = [
        os.path.join(test_photo_dir, f).replace("\\", "/")
        for f in sorted(os.listdir(test_photo_dir))
        if f.lower().endswith(valid_exts)
    ]
    if not test_paths:
        raise FileNotFoundError(f"❌ 在 '{test_photo_dir}' 找不到測試圖片，請確認已執行 split_dataset.py！")

    filename_to_path = {os.path.basename(p): p for p in test_paths}

    if os.path.exists(preview_root):
        shutil.rmtree(preview_root)
    os.makedirs(preview_root, exist_ok=True)

    # 1. 分別用「微調前」與「微調後」模型一次性撈出所有題目的候選圖
    base_candidates = query_model_candidates(base_ckpt, "原始 ImageBind", test_paths, pooling_tasks, top_k_per_query, device)
    tuned_candidates = query_model_candidates(tuned_ckpt, "微調後 ImageBind", test_paths, pooling_tasks, top_k_per_query, device)

    # 2. 合併去重後，改以檔名排序，方便人工檢查與對照
    draft_queries = []
    for task in pooling_tasks:
        print(f"正在處理題目: {task['query_id']} - {task['activity_name']}")
        q_id = task["query_id"]
        act_name = task["activity_name"]
        main_desc = task["main_description"]

        hit_counter = Counter()
        for f in base_candidates[q_id]:
            hit_counter[f] += 1
        for f in tuned_candidates[q_id]:
            hit_counter[f] += 1

        sorted_pool = sorted(hit_counter.keys())
        print(f"📌 [{q_id}] {act_name}：合併雙模型去重後共 {len(sorted_pool)} 張候選測試圖（依檔名排序）。")

        task_dir = os.path.join(preview_root, f"{q_id}_{act_name}")
        os.makedirs(task_dir, exist_ok=True)
        for fname in sorted_pool:
            shutil.copy(filename_to_path[fname], os.path.join(task_dir, fname))

        draft_queries.append({
            "query_id": q_id,
            "activity_name": act_name,
            "description": main_desc,
            "relevance": {fname: 0 for fname in sorted_pool}
        })

    os.makedirs(os.path.dirname(draft_json_path), exist_ok=True)
    with open(draft_json_path, "w", encoding="utf-8") as f:
        json.dump(draft_queries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("🎉 雙模型公平候選池建立完成！接下來請進行人工打分：")
    print(f"1. 打開資料夾看縮圖: {preview_root}")
    print(f"2. 編輯草稿 JSON 檔案: {draft_json_path}")
    print("   👉 符合活動的照片請將 0 改為 2 (完美符合) 或 1 (部分符合)")
    print("   👉 誤判的照片維持 0 (或刪除)，完成後另存為 ../data/eval_queries.json 即可！")
    print("=" * 72)

# ==========================================
# 主程式進入點 (Main Entry)
# ==========================================
if __name__ == "__main__":
    sample_pooling_tasks = [
        {
            "query_id": "q_01",
            "activity_name": "校慶運動會",
            "main_description": "students running on a red athletics track during a school sports day",
            "pooling_prompts": [
                "students running on a red athletics track during a school sports day",
                "high school track and field relay race competition in an outdoor stadium",
                "students cheering and holding medals at a school sports meet"
            ]
        },
        {
            "query_id": "q_02",
            "activity_name": "科學實驗室活動",
            "main_description": "students conducting chemistry experiments with microscopes in a science laboratory",
            "pooling_prompts": [
                "students conducting chemistry experiments with microscopes in a science laboratory",
                "high school STEM robotics competition with students testing robots",
                "students in white lab coats performing biology experiments in a lab"
            ]
        },
        {
            "query_id": "q_03",
            "activity_name": "校園音樂會",
            "main_description": "school orchestra playing violins and musical instruments on stage",
            "pooling_prompts": [
                "school orchestra playing violins and musical instruments on stage",
                "students choir singing together in uniforms during a school music concert",
                "student band performing with guitars and drums at a school talent show"
            ]
        },
        {
            "query_id": "q_04",
            "activity_name": "畢業典禮與頒獎",
            "main_description": "students wearing graduation gowns and caps on stage during graduation ceremony",
            "pooling_prompts": [
                "students wearing graduation gowns and caps on stage during graduation ceremony",
                "school principal shaking hands and presenting awards or trophies to students",
                "graduating class taking a formal group photo on stage in academic gowns"
            ]
        },
        {
            "query_id": "q_05",
            "activity_name": "球類運動與體育館",
            "main_description": "students playing basketball or badminton inside a school indoor gymnasium",
            "pooling_prompts": [
                "students playing basketball or badminton inside a school indoor gymnasium",
                "high school volleyball or table tennis match in an indoor sports hall",
                "students practicing football on a school outdoor turf pitch"
            ]
        }
    ]

    build_fair_eval_pool(sample_pooling_tasks, top_k_per_query=10)