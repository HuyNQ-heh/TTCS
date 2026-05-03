import numpy as np
from model_TFIDF import df, X, top_k_nearest_neighbors

# =========================
# 1. Định nghĩa relevant
#    Ở đây dùng genre làm ground truth
# =========================
def is_relevant(query_idx, candidate_idx):
    return df.loc[query_idx, "genre"] == df.loc[candidate_idx, "genre"]

# =========================
# 2. Precision@K cho 1 query
# =========================
def precision_at_k(query_idx, neighbors, k):
    relevant_count = 0

    for idx, _ in neighbors[:k]:
        if is_relevant(query_idx, idx):
            relevant_count += 1

    return relevant_count / k

# =========================
# 3. Đánh giá toàn bộ model
# =========================
def evaluate_model(k=5, num_samples=100):
    if num_samples > len(df):
        num_samples = len(df)

    indices = np.random.choice(len(df), size=num_samples, replace=False)

    scores = []

    for query_idx in indices:
        query_vector = X[query_idx]

        neighbors = top_k_nearest_neighbors(
            query_vector=query_vector,
            data_matrix=X,
            k=k,
            exclude_index=query_idx
        )

        score = precision_at_k(query_idx, neighbors, k)
        scores.append(score)

    return np.mean(scores)

# =========================
# 4. Chạy thử
# =========================
if __name__ == "__main__":
    p_at_5 = evaluate_model(k=5, num_samples=100)
    p_at_10 = evaluate_model(k=10, num_samples=100)

    print("Precision@5:", round(p_at_5, 4))
    print("Precision@10:", round(p_at_10, 4))