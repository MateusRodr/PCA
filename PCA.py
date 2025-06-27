from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

# Dados de exemplo: carregando a base Iris
X, y = load_iris(return_X_y=True)

# Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM sem PCA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(f"Acurácia SVM sem PCA: {accuracy_score(y_test, y_pred):.4f}")

# SVM com PCA
pca = PCA(n_components=0.95)  # mantém 95% da variância
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
svm_pca = SVC(kernel='rbf')
svm_pca.fit(X_train_pca, y_train)
y_pred_pca = svm_pca.predict(X_test_pca)
print(f"Acurácia SVM com PCA: {accuracy_score(y_test, y_pred_pca):.4f}")

# K-means sem PCA
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
print(f"Silhouette Score sem PCA: {silhouette_score(X_scaled, labels):.4f}")

# K-means com PCA
kmeans_pca = KMeans(n_clusters=3, random_state=42)
kmeans_pca.fit(X_pca)
labels_pca = kmeans_pca.labels_
print(f"Silhouette Score com PCA: {silhouette_score(X_pca, labels_pca):.4f}")
