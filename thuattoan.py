import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. Đọc dữ liệu
df = pd.read_csv(r"D:\archive\Music Info.csv")

# 2. Loại trùng
df = df.drop_duplicates(subset=["spotify_id"]).reset_index(drop=True)

# 3. Chọn các đặc trưng số cho content-based
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

# 4. Loại các dòng bị thiếu ở cột đặc trưng
df = df.dropna(subset=feature_cols).reset_index(drop=True)

# 5. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols])

# 6. Xây dựng mô hình tìm láng giềng gần nhất
# Dùng cosine để giống ý tưởng cosine similarity nhưng không tạo full matrix
model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
model.fit(X)


# 7. Hàm gợi ý bài hát
def recommend_songs(song_name, top_n=10):
    # tìm các bài có tên khớp
    matches = df[df["name"].str.lower() == song_name.lower()]

    if matches.empty:
        return f"Không tìm thấy bài hát: {song_name}"

    # lấy bài đầu tiên nếu có nhiều bản trùng tên
    idx = matches.index[0]

    # tìm hàng xóm gần nhất
    distances, indices = model.kneighbors([X[idx]], n_neighbors=top_n + 1)

    rec_indices = indices[0][1:]  # bỏ chính nó
    rec_distances = distances[0][1:]

    results = df.loc[rec_indices, ["name", "artist", "genre", "year"]].copy()
    results["distance"] = rec_distances
    results["similarity_score"] = 1 - results["distance"]

    return results.reset_index(drop=True)


# 8. Demo
print(recommend_songs("Mr. Brightside", top_n=10))