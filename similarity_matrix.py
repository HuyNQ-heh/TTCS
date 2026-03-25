import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"D:\archive\Music Info.csv")

features = [
'danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness',
'valence','tempo'
]
# X=df[features]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=features,index=df['name'])
# svd = TruncatedSVD(n_components=3)
# X_reduced = svd.fit_transform(X_scaled)
# similarity_matrix = cosine_similarity(X_reduced)
# sim_ma_dense=pd.DataFrame(similarity_matrix,columns=X_reduced['name'],index=X_reduced['name'])
df_sample=df.sample(2000)
X = df_sample[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features,index=df_sample['name'])
similarity_matrix = cosine_similarity(X_scaled)
sim_ma_dense=pd.DataFrame(similarity_matrix,columns=df_sample['name'],index=df_sample['name'])
sim_ma_dense.to_csv("similarity_matrix.csv")