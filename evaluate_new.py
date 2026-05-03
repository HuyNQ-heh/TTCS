import numpy as np
import pandas as pd
from model import df, X, top_k_nearest_neighbors_multiple_exclude

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
# 2. Lấy tập tag đại diện cho 5 bài đã nghe
# =========================
def get_recent_tags(recent_indices):
    recent_tags = set()

    for idx in recent_indices:
        recent_tags |= get_tag_set(idx)

    return recent_tags


# =========================
# 3. Tính Jaccard giữa tập tag history và bài candidate
# =========================
def jaccard_similarity_recent(recent_indices, candidate_idx):
    recent_tags = get_recent_tags(recent_indices)
    candidate_tags = get_tag_set(candidate_idx)

    if len(recent_tags) == 0 or len(candidate_tags) == 0:
        return 0.0

    intersection = len(recent_tags & candidate_tags)
    union = len(recent_tags | candidate_tags)

    return intersection / union


# =========================
# 4. Kiểm tra candidate có relevant không
# =========================
def is_relevant_recent(recent_indices, candidate_idx, threshold=0.3):
    return jaccard_similarity_recent(recent_indices, candidate_idx) >= threshold


# =========================
# 5. Precision@K cho 1 nhóm 5 bài đã nghe
# =========================
def precision_at_k_recent(recent_indices, neighbors, k, threshold=0.3):
    relevant_count = 0

    for idx, _ in neighbors[:k]:
        if is_relevant_recent(recent_indices, idx, threshold):
            relevant_count += 1

    return relevant_count / k, relevant_count


# =========================
# 6. Lọc bài hợp lệ: chỉ lấy bài có tags
# =========================
def get_valid_indices():
    valid_indices = []

    for i in range(len(df)):
        tags = df.loc[i, "tags"]
        if pd.notna(tags) and str(tags).strip() != "":
            valid_indices.append(i)

    return valid_indices


# =========================
# 7. Đánh giá model theo 5 bài gần nhất
# =========================
def evaluate_recent_model_detail(
    k_list=[1, 3, 5, 10],
    num_samples=100,
    threshold=0.3,
    history_size=5
):
    valid_indices = get_valid_indices()

    if len(valid_indices) < history_size:
        raise ValueError("Không đủ bài hát có tags để tạo lịch sử nghe.")

    max_k = max(k_list)

    precision_scores = {k: [] for k in k_list}
    relevant_counts = {k: [] for k in k_list}

    query_details = []

    for sample_id in range(num_samples):
        # Chọn ngẫu nhiên 5 bài làm lịch sử nghe gần nhất
        recent_indices = np.random.choice(
            valid_indices,
            size=history_size,
            replace=False
        )

        # Vector đại diện cho sở thích = trung bình vector của 5 bài
        query_vector = np.mean(X[recent_indices], axis=0)

        # Tìm top K bài gần nhất, loại bỏ 5 bài đã nghe
        neighbors = top_k_nearest_neighbors_multiple_exclude(
            query_vector=query_vector,
            data_matrix=X,
            k=max_k,
            exclude_indices=recent_indices
        )

        for k in k_list:
            p_at_k, rel_count = precision_at_k_recent(
                recent_indices=recent_indices,
                neighbors=neighbors,
                k=k,
                threshold=threshold
            )

            precision_scores[k].append(p_at_k)
            relevant_counts[k].append(rel_count)

        recent_song_names = []
        for idx in recent_indices:
            name = df.loc[idx, "name"] if "name" in df.columns else ""
            artist = df.loc[idx, "artist"] if "artist" in df.columns else ""
            recent_song_names.append(f"{name} - {artist}")

        query_details.append({
            "sample_id": sample_id + 1,
            "recent_songs": " | ".join(recent_song_names),
            "recent_tags_count": len(get_recent_tags(recent_indices)),
            **{f"precision@{k}": precision_scores[k][-1] for k in k_list},
            **{f"relevant_count@{k}": relevant_counts[k][-1] for k in k_list}
        })

    summary = {
        "num_samples": num_samples,
        "threshold": threshold,
        "history_size": history_size,
        "valid_songs": len(valid_indices),
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
# 8. In báo cáo
# =========================
def print_evaluation_report(summary):
    print("=" * 60)
    print("EVALUATION REPORT")
    print("Model        : Recent 5 songs -> Average vector -> KNN")
    print("Ground truth : Tags (Jaccard similarity)")
    print(f"Valid songs  : {summary['valid_songs']}")
    print(f"Samples used : {summary['num_samples']}")
    print(f"History size : {summary['history_size']} songs")
    print(f"Threshold    : {summary['threshold']}")
    print("=" * 60)

    for k, values in summary["metrics"].items():
        print(
            f"Precision@{k:<2}: {values['precision_mean']:.4f} "
            f"| Std: {values['precision_std']:.4f}"
        )
        print(f"Avg relevant in top {k}: {values['avg_relevant_count']:.2f}")
        print("-" * 60)


# =========================
# 9. Đánh giá nhiều threshold
# =========================
def evaluate_multiple_thresholds(
    thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
    num_samples=100,
    history_size=5
):
    rows = []

    for th in thresholds:
        summary, _ = evaluate_recent_model_detail(
            k_list=[1, 3, 5, 10],
            num_samples=num_samples,
            threshold=th,
            history_size=history_size
        )

        row = {
            "threshold": th,
            "num_samples": summary["num_samples"],
            "history_size": summary["history_size"],
            "Precision@1": summary["metrics"][1]["precision_mean"],
            "Precision@3": summary["metrics"][3]["precision_mean"],
            "Precision@5": summary["metrics"][5]["precision_mean"],
            "Precision@10": summary["metrics"][10]["precision_mean"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# 10. Main
# =========================
if __name__ == "__main__":
    summary, details_df = evaluate_recent_model_detail(
        k_list=[1, 3, 5, 10],
        num_samples=100,
        threshold=0.3,
        history_size=5
    )

    print_evaluation_report(summary)

    details_df.to_csv("evaluation_recent_details.csv", index=False)

    threshold_df = evaluate_multiple_thresholds(
        thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
        num_samples=100,
        history_size=5
    )

    print("\nTHRESHOLD COMPARISON")
    print(threshold_df.to_string(index=False))

    threshold_df.to_csv("evaluation_recent_threshold_comparison.csv", index=False)