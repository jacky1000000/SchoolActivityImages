import os
import json
import math
import torch
import torch.nn.functional as F
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data
import shutil

def compute_ap(ranked_filenames, relevance_dict):
    """計算單一查詢的 Average Precision (AP)"""
    relevant_set = {fname for fname, score in relevance_dict.items() if score > 0}
    if not relevant_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank_idx, fname in enumerate(ranked_filenames, start=1):
        if fname in relevant_set:
            hits += 1
            sum_precisions += (hits / rank_idx)

    return sum_precisions / len(relevant_set)

def compute_ndcg(ranked_filenames, relevance_dict, k=10):
    """計算單一查詢在 Top-K 的 Normalized Discounted Cumulative Gain (nDCG@K)"""
    dcg = 0.0
    for i, fname in enumerate(ranked_filenames[:k], start=1):
        rel = relevance_dict.get(fname, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)

    ideal_rels = sorted([score for score in relevance_dict.values() if score > 0], reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))

    return (dcg / idcg) if idcg > 0 else 0.0

def evaluate_single_model(model_name, checkpoint_path, test_photo_dir, eval_json_path, device="cuda", batch_size=32):
    """載入指定權重，對 test_photos/ 建立索引並計算 mAP、nDCG@5 與 nDCG@10"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 找不到權重檔案 '{checkpoint_path}'，跳過 [{model_name}] 評估。")
        return None

    print(f"\n⏳ 正在載入 [{model_name}] 並提取測試集特徵...")
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    # 1. 提取 test_photos/ 內所有測試相片的 1024 維向量
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    test_paths = [
        os.path.join(test_photo_dir, f).replace("\\", "/")
        for f in sorted(os.listdir(test_photo_dir))
        if f.lower().endswith(valid_exts)
    ]
    test_filenames = [os.path.basename(p) for p in test_paths]

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(test_paths), batch_size):
            batch = test_paths[i:i + batch_size]
            inputs = {ModalityType.VISION: imagebind_data.load_and_transform_vision_data(batch, device)}
            v_embeds = F.normalize(model(inputs)[ModalityType.VISION], p=2, dim=-1)
            all_embeds.append(v_embeds)
    photo_matrix = torch.cat(all_embeds, dim=0)

    # 2. 讀取人工標記的標準答案測試集並逐題評估
    with open(eval_json_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    ap_scores, ndcg5_scores, ndcg10_scores = [], [], []

    for q in queries:
        desc = q["description"]
        rel_dict = q["relevance"]

        with torch.no_grad():
            text_inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text([desc], device)}
            text_embed = F.normalize(model(text_inputs)[ModalityType.TEXT], p=2, dim=-1)

        similarities = torch.matmul(photo_matrix, text_embed.T).squeeze(-1)
        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        ranked_filenames = [test_filenames[idx] for idx in sorted_indices]
        print("First 10 ranked filenames:", ranked_filenames[:10])
        # check if using base or tuned model
        if "Original" in model_name:
            result_folder = "../../output/MatchingResultsObtained/base"
        else:
            result_folder = "../../output/MatchingResultsObtained/tuned"
        # create folder for the query, the name should be query_id
        query_folder = os.path.join(result_folder, q["query_id"])
        os.makedirs(query_folder, exist_ok=True)
        # copy the first 10 ranked photos to the query folder
        for rank, filename in enumerate(ranked_filenames[:10], start=1):
            src_path = os.path.join(test_photo_dir, filename)
            dst_path = os.path.join(query_folder, f"Top {rank}_{filename}")
            shutil.copy(src_path, dst_path)
        ap_scores.append(compute_ap(ranked_filenames, rel_dict))
        ndcg5_scores.append(compute_ndcg(ranked_filenames, rel_dict, k=5))
        ndcg10_scores.append(compute_ndcg(ranked_filenames, rel_dict, k=10))

    # 釋放顯存供下一個模型使用
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "mAP": sum(ap_scores) / len(ap_scores),
        "nDCG@5": sum(ndcg5_scores) / len(ndcg5_scores),
        "nDCG@10": sum(ndcg10_scores) / len(ndcg10_scores)
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_photos_folder = "../../data/test_photos"
    eval_queries_file = "../../data/eval_queries_draft_test_data.json"

    base_ckpt = "../../my_checkpoints/imagebind_huge.pth"
    tuned_ckpt = "../../ckpt_updated/imagebind_campus_tuned/imagebind_campus_finetuned.pth"

    os.makedirs("../../output/MatchingResultsObtained", exist_ok=True)
    # make 2 folders for storing results of base and tuned models
    os.makedirs("../../output/MatchingResultsObtained/base", exist_ok=True)
    os.makedirs("../../output/MatchingResultsObtained/tuned", exist_ok=True)

    # 分別評估微調前與微調後模型
    base_res = evaluate_single_model("1. Original ImageBind (Zero-Shot)", base_ckpt, test_photos_folder, eval_queries_file, device)
    tuned_res = evaluate_single_model("2. Fine-tuned ImageBind (BLIP-Distilled)", tuned_ckpt, test_photos_folder, eval_queries_file, device)

    # 列印最終對比報告
    print("\n" + "=" * 76)
    print("🏆 校園活動相片檢索（測試集 N=270）：mAP 與 nDCG 量化評估對比報告")
    print("=" * 76)
    print(f"{'模型版本 (Model Version)':<38} | {'mAP':<10} | {'nDCG@5':<10} | {'nDCG@10':<10}")
    print("-" * 76)

    if base_res:
        print(f"{'1. Original ImageBind (Not Trained)':<38} | {base_res['mAP']:<10.4f} | {base_res['nDCG@5']:<10.4f} | {base_res['nDCG@10']:<10.4f}")
    if tuned_res:
        print(f"{'2. Fine-tuned ImageBind (Trained)':<38} | {tuned_res['mAP']:<10.4f} | {tuned_res['nDCG@5']:<10.4f} | {tuned_res['nDCG@10']:<10.4f}")

    if base_res and tuned_res:
        print("-" * 76)
        diff_map = (tuned_res['mAP'] - base_res['mAP']) * 100
        diff_ndcg5 = (tuned_res['nDCG@5'] - base_res['nDCG@5']) * 100
        diff_ndcg10 = (tuned_res['nDCG@10'] - base_res['nDCG@10']) * 100
        print(f"{'📈 絕對提升幅度 (Gain)':<38} | {diff_map:+.2f}%    | {diff_ndcg5:+.2f}%    | {diff_ndcg10:+.2f}%")
    print("=" * 76)
