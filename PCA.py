import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dados
X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# K-means sem PCA (para rotular)
kmeans_full = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
labels_full = kmeans_full.labels_

# PCA para visualização
pca_vis = PCA(n_components=2)
X_2D = pca_vis.fit_transform(X_scaled)

# K-means com PCA (para rotular)
kmeans_pca = KMeans(n_clusters=3, random_state=42).fit(X_2D)
labels_pca = kmeans_pca.labels_

# Plot
plt.figure(figsize=(12, 5))

# Sem PCA
plt.subplot(1, 2, 1)
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels_full, cmap='Set1', s=50)
plt.title("K-means (sem PCA) - Visualizado em 2D")

# Com PCA
plt.subplot(1, 2, 2)
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels_pca, cmap='Set2', s=50)
plt.title("K-means (com PCA 2D)")

plt.tight_layout()
plt.show()
