import numpy as np
import pandas as pd
from model_oh import df, X, top_k_nearest_neighbors

np.random.seed(42)


# =========================
# 1. Lấy tập tag của 1 bài hát
# =========================
def get_tag_set(idx):
    tags = df.loc[idx, "tags"]

    if pd.isna(tags) or str(tags).strip() == "":
        return set()

    return set(tag.strip().lower() for tag in str(tags).split(",") if tag.strip())


# =========================
# 2. Tính Jaccard similarity giữa tags của 2 bài
# =========================
def jaccard_similarity_tags(query_idx, candidate_idx):
    query_tags = get_tag_set(query_idx)
    candidate_tags = get_tag_set(candidate_idx)

    if len(query_tags) == 0 and len(candidate_tags) == 0:
        return 1.0
    if len(query_tags) == 0 or len(candidate_tags) == 0:
        return 0.0

    intersection = len(query_tags & candidate_tags)
    union = len(query_tags | candidate_tags)

    return intersection / union


# =========================
# 3. Kiểm tra 1 bài có relevant không
# =========================
def is_relevant(query_idx, candidate_idx, threshold=0.3):
    return jaccard_similarity_tags(query_idx, candidate_idx) >= threshold


# =========================
# 4. Tính Precision@K cho 1 query
# =========================
def precision_at_k(query_idx, neighbors, k, threshold=0.3):
    relevant_count = 0

    for idx, _ in neighbors[:k]:
        if is_relevant(query_idx, idx, threshold):
            relevant_count += 1

    return relevant_count / k, relevant_count


# =========================
# 5. Lọc các query hợp lệ
#    chỉ giữ bài có tags
# =========================
def get_valid_query_indices():
    valid_indices = []

    for i in range(len(df)):
        tags = df.loc[i, "tags"]
        if pd.notna(tags) and str(tags).strip() != "":
            valid_indices.append(i)

    return valid_indices


# =========================
# 6. Đánh giá model chi tiết
# =========================
def evaluate_model_detail(k_list=[1, 3, 5, 10], num_samples=100, threshold=0.3):
    valid_indices = get_valid_query_indices()

    if len(valid_indices) == 0:
        raise ValueError("Không có bài hát nào có tags để evaluate.")

    if num_samples > len(valid_indices):
        num_samples = len(valid_indices)

    sampled_indices = np.random.choice(valid_indices, size=num_samples, replace=False)

    max_k = max(k_list)

    precision_scores = {k: [] for k in k_list}
    relevant_counts = {k: [] for k in k_list}

    # lưu thêm chi tiết từng query nếu muốn xuất CSV
    query_details = []

    for query_idx in sampled_indices:
        query_vector = X[query_idx]

        neighbors = top_k_nearest_neighbors(
            query_vector=query_vector,
            data_matrix=X,
            k=max_k,
            exclude_index=query_idx
        )

        for k in k_list:
            p_at_k, rel_count = precision_at_k(query_idx, neighbors, k, threshold)
            precision_scores[k].append(p_at_k)
            relevant_counts[k].append(rel_count)

        query_details.append({
            "query_index": query_idx,
            "query_name": df.loc[query_idx, "name"] if "name" in df.columns else "",
            "query_artist": df.loc[query_idx, "artist"] if "artist" in df.columns else "",
            **{f"precision@{k}": precision_scores[k][-1] for k in k_list},
            **{f"relevant_count@{k}": relevant_counts[k][-1] for k in k_list}
        })

    summary = {
        "num_samples": num_samples,
        "threshold": threshold,
        "valid_queries": len(valid_indices),
        "metrics": {}
    }

    for k in k_list:
        summary["metrics"][k] = {
            "precision_mean": float(np.mean(precision_scores[k])),
            "precision_std": float(np.std(precision_scores[k])),
            "avg_relevant_count": float(np.mean(relevant_counts[k]))
        }

    details_df = pd.DataFrame(query_details)
    return summary, details_df


# =========================
# 7. In báo cáo
# =========================
def print_evaluation_report(summary):
    print("=" * 50)
    print("EVALUATION REPORT")
    print("Ground truth : Tags (Jaccard similarity)")
    print(f"Valid queries: {summary['valid_queries']}")
    print(f"Samples used : {summary['num_samples']}")
    print(f"Threshold    : {summary['threshold']}")
    print("=" * 50)

    for k, values in summary["metrics"].items():
        print(f"Precision@{k:<2}: {values['precision_mean']:.4f}  |  Std: {values['precision_std']:.4f}")
        print(f"Avg relevant in top {k}: {values['avg_relevant_count']:.2f}")
        print("-" * 50)


# =========================
# 8. Chạy nhiều threshold trong 1 lần
# =========================
def evaluate_multiple_thresholds(thresholds=[0.1, 0.15, 0.2, 0.25, 0.3], num_samples=100):
    rows = []

    for th in thresholds:
        summary, _ = evaluate_model_detail(
            k_list=[1, 3, 5, 10],
            num_samples=num_samples,
            threshold=th
        )

        row = {
            "threshold": th,
            "num_samples": summary["num_samples"],
            "Precision@1": summary["metrics"][1]["precision_mean"],
            "Precision@3": summary["metrics"][3]["precision_mean"],
            "Precision@5": summary["metrics"][5]["precision_mean"],
            "Precision@10": summary["metrics"][10]["precision_mean"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# 9. Main
# =========================
if __name__ == "__main__":
    # Báo cáo chi tiết cho 1 threshold
    summary, details_df = evaluate_model_detail(
        k_list=[1, 3, 5, 10],
        num_samples=100,
        threshold=0.3
    )

    print_evaluation_report(summary)

    # Lưu chi tiết từng query
    details_df.to_csv("evaluation_details_tags.csv", index=False)

    # Bảng so sánh nhiều threshold
    threshold_df = evaluate_multiple_thresholds(
        thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
        num_samples=100
    )

    print("\nTHRESHOLD COMPARISON")
    print(threshold_df.to_string(index=False))

    threshold_df.to_csv("evaluation_threshold_comparison.csv", index=False)