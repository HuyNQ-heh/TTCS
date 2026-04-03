import pandas as pd
import numpy as np

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv(r"D:\archive\Music Info.csv")

# =========================
# 2. Loại bỏ trùng lặp
# =========================
if "spotify_id" in df.columns:
    df = df.drop_duplicates(subset=["spotify_id"]).reset_index(drop=True)
else:
    df = df.drop_duplicates(subset=["name", "artist"]).reset_index(drop=True)

# =========================
# 3. Chọn các cột đặc trưng
# =========================
feature_cols = [
    "year",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]

# chỉ giữ các cột thật sự tồn tại
feature_cols = [col for col in feature_cols if col in df.columns]

# bỏ dòng thiếu dữ liệu
df = df.dropna(subset=feature_cols).reset_index(drop=True)

# =========================
# 4. Chuẩn hóa dữ liệu thủ công
# =========================
X_raw = df[feature_cols].values.astype(np.float32)

mean = X_raw.mean(axis=0)
std = X_raw.std(axis=0)
std[std == 0] = 1.0

X = (X_raw - mean) / std

# =========================
# 5. Hàm cosine similarity giữa 2 vector
# =========================
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)

# =========================
# 6. Hàm cosine distance giữa 2 vector
#    distance = 1 - similarity
# =========================
def cosine_distance(vec1, vec2):
    return 1 - cosine_similarity(vec1, vec2)

# =========================
# 7. Hàm nearest neighbor
#    Trả về 1 hàng xóm gần nhất
# =========================
def nearest_neighbor(query_vector, data_matrix):
    best_index = -1
    best_distance = float("inf")

    for i in range(len(data_matrix)):
        dist = cosine_distance(query_vector, data_matrix[i])

        if dist < best_distance:
            best_distance = dist
            best_index = i

    return best_index, best_distance

# =========================
# 8. Hàm top-k nearest neighbors
# =========================
def top_k_nearest_neighbors(query_vector, data_matrix, k=10, exclude_index=None):
    distances = []

    for i in range(len(data_matrix)):
        if exclude_index is not None and i == exclude_index:
            continue

        dist = cosine_distance(query_vector, data_matrix[i])
        distances.append((i, dist))

    # sắp xếp tăng dần theo distance
    distances.sort(key=lambda x: x[1])

    return distances[:k]

# =========================
# 9. Tìm index bài hát theo tên / ca sĩ
# =========================
def find_song_index(song_name, artist_name=None):
    temp = df[df["name"].str.lower() == song_name.lower()]

    if artist_name is not None and "artist" in df.columns:
        temp = temp[temp["artist"].str.lower() == artist_name.lower()]

    if temp.empty:
        return None

    return temp.index[0]

# =========================
# 10. Hàm recommend từ nearest neighbors tự code
# =========================
def recommend_songs(song_name, artist_name=None, top_n=10):
    song_idx = find_song_index(song_name, artist_name)

    if song_idx is None:
        return f"Không tìm thấy bài hát: {song_name}"

    query_vector = X[song_idx]

    neighbors = top_k_nearest_neighbors(
        query_vector=query_vector,
        data_matrix=X,
        k=top_n,
        exclude_index=song_idx
    )

    neighbor_indices = [idx for idx, dist in neighbors]
    neighbor_distances = [dist for idx, dist in neighbors]
    similarity_scores = [1 - dist for dist in neighbor_distances]

    result_cols = [col for col in ["name", "artist", "genre", "year"] if col in df.columns]
    results = df.loc[neighbor_indices, result_cols].copy()
    results["distance"] = neighbor_distances
    results["similarity_score"] = similarity_scores

    return results.reset_index(drop=True)

# =========================
# 11. Demo
# =========================
print(recommend_songs("Mr. Brightside", top_n=10))