import os
import json
import shutil
import torch
import torch.nn.functional as F
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data

def build_pooling_dataset(
    pooling_tasks,
    cache_path="../../data/test_photo_embeddings.pt",
    checkpoint_path="../../my_checkpoints/imagebind_huge.pth",
    preview_root="../../output/pooling_preview",
    draft_json_path="../../data/eval_queries_draft.json",
    top_k_per_query=20
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"❌ 找不到特徵快取 '{cache_path}'，請先執行一次 quick_search.py 建立 1,330 張相片索引！")

    # 1. 載入快取的 1,330 張相片特徵與路徑
    cache = torch.load(cache_path, map_location=device)
    all_paths = cache["paths"]
    all_filenames = [os.path.basename(p) for p in all_paths]
    filename_to_path = dict(zip(all_filenames, all_paths))
    photo_embeddings = cache["embeddings"].to(device)

    # 2. 載入 ImageBind 編碼查詢文字
    print("⏳ 正在載入 ImageBind 模型進行多重提示詞池化檢索...")
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    if os.path.exists(preview_root):
        shutil.rmtree(preview_root)
    os.makedirs(preview_root, exist_ok=True)

    draft_queries = []

    for task in pooling_tasks:
        q_id = task["query_id"]
        act_name = task["activity_name"]
        main_desc = task["main_description"]
        sub_queries = task["pooling_prompts"]

        print(f"\n📌 正在為 [{q_id}] {act_name} 建立候選池...")
        pooled_filenames = set()

        # 使用多組不同說法的提示詞各自檢索 Top-K 並合併去重
        for sq in sub_queries:
            with torch.no_grad():
                text_inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text([sq], device)}
                text_embed = F.normalize(model(text_inputs)[ModalityType.TEXT], p=2, dim=-1)

            similarities = torch.matmul(photo_embeddings, text_embed.T).squeeze(-1)
            _, top_indices = torch.topk(similarities, k=min(top_k_per_query, len(all_paths)))

            for idx in top_indices.tolist():
                pooled_filenames.add(all_filenames[idx])

        sorted_pool = sorted(list(pooled_filenames))
        print(f"   ✅ 合併去重後共撈出 {len(sorted_pool)} 張候選相片。")

        # 將這批候選相片複製到預覽資料夾，方便您直接打開 Windows 檔案總管看圖審核
        task_preview_dir = os.path.join(preview_root, f"{q_id}_{act_name}")
        os.makedirs(task_preview_dir, exist_ok=True)
        for fname in sorted_pool:
            shutil.copy(filename_to_path[fname], os.path.join(task_preview_dir, fname))

        # 建立預填 JSON 草稿（預設全部給 0 分，等您人工看圖後把正確的改成 1 或 2）
        relevance_draft = {fname: 0 for fname in sorted_pool}
        draft_queries.append({
            "query_id": q_id,
            "activity_name": act_name,
            "description": main_desc,
            "relevance": relevance_draft
        })

    # 儲存草稿 JSON
    os.makedirs(os.path.dirname(draft_json_path), exist_ok=True)
    with open(draft_json_path, "w", encoding="utf-8") as f:
        json.dump(draft_queries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("🎉 多重查詢池化完成！接下來請進行人工審核：")
    print(f"1. 打開預覽資料夾檢視縮圖: {preview_root}")
    print(f"2. 打開草稿檔案進行打分:   {draft_json_path}")
    print("   👉 看到真正符合該活動的照片，將分數改為 2 (完美符合) 或 1 (部分符合)")
    print("   👉 誤判無關的照片維持 0 (或直接刪除該行)，完成後另存為 eval_queries.json 即可！")
    print("=" * 72)

if __name__ == "__main__":
    # 在這裡設定您想評估的校園活動與多角度池化提示詞
    sample_pooling_tasks = [
        {
            "query_id": "q_01",
            "activity_name": "校慶運動會",
            "main_description": "students running on a red athletics track during a school sports day",
            "pooling_prompts": [
                "students running on a red athletics track during a school sports day",
                "school track and field relay race competition outdoor stadium",
                "students cheering and holding medals at a school sports meet"
            ]
        },
        {
            "query_id": "q_02",
            "activity_name": "科學實驗室活動",
            "main_description": "students conducting chemistry experiments with microscopes in a science laboratory",
            "pooling_prompts": [
                "students conducting chemistry experiments with microscopes in a science laboratory",
                "high school biology lab class with students in white lab coats",
                "STEM robotics and science exhibition booth at school"
            ]
        },
        {
            "query_id": "q_03",
            "activity_name": "校園音樂會",
            "main_description": "school orchestra playing violins and musical instruments on stage",
            "pooling_prompts": [
                "school orchestra playing violins and musical instruments on stage",
                "students choir singing and performing in school auditorium",
                "band performance with drums and guitars at school talent show"
            ]
        },
        {
            "query_id": "q_04",
            "activity_name": "畢業典禮",
            "main_description": "students wearing graduation gowns and caps on stage during graduation ceremony",
            "pooling_prompts": [
                "students wearing graduation gowns and caps on stage during graduation ceremony",
                "principal shaking hands with graduating student holding diploma",
                "group photo of graduating class throwing caps in the air"
            ]
        }
    ]

    build_pooling_dataset(sample_pooling_tasks, top_k_per_query=20)
