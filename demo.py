import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv(r"D:\archive\Music Info.csv")



features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]
df = df.dropna(subset=features)
scaler = StandardScaler()
X = scaler.fit_transform(df[features]) 
sc=cosine_similarity(X)
# def recommend(song_name, top_n):
#
#     if song_name not in df['name'].values:
#         print("Không tìm thấy bài hát")
#         return
#
#     index = df[df['name'] == song_name].index[0]
#
#     song_vector = X[index].reshape(1, -1)
#
#     similarity_scores = cosine_similarity(song_vector, X)[0]
#
#     similarity_scores = list(enumerate(similarity_scores))
#
#     similarity_scores = sorted(
#         similarity_scores,
#         key=lambda x: x[1],
#         reverse=True
#     )
#
#     top_songs = similarity_scores[1:top_n+1]
#
#     print(f"\nGợi ý cho bài: {song_name}\n")
#
#     for i, score in top_songs:
#
#         print(
#             df.iloc[i]['name'],
#             "-",
#             df.iloc[i]['artist'],
#             "Similarity:",
#             round(score,3)
#         )
#
# recommend("Mr. Brightside", 5)