import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

# Load Dataset
file_path = 'Customers.csv'
customers_data = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
# Menangani Data yang Hilang
missing_values = customers_data.isnull().sum()

# Analisis Distribusi Variabel Numerik
numerical_columns = customers_data.select_dtypes(include=['int64', 'float64']).columns
numerical_summary = customers_data[numerical_columns].describe()

# Visualisasi Data: Distribusi Variabel Numerik
plt.figure(figsize=(12, 8))
customers_data[numerical_columns].hist(bins=15, edgecolor='black', figsize=(15, 10))
plt.tight_layout()
plt.show()

# Analisis Korelasi
correlation_matrix = customers_data[numerical_columns].corr()

# Visualisasi Data: Heatmap untuk Matriks Korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap Korelasi Variabel Numerik")
plt.show()

# Visualisasi Variabel Kategorikal
categorical_columns = customers_data.select_dtypes(include=['object']).columns

plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    customers_data[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Distribusi {col}")
plt.tight_layout()
plt.show()

# Menampilkan hasil analisis
print("Data Information:")
print(customers_data.info())

print("\nMissing Values:")
print(missing_values)

print("\nSummary Statistics for Numerical Variables:")
print(numerical_summary)

# Data Preprocessing
# Menghapus Data Kosong (Missing Values)
customers_data = customers_data.dropna()

# Menghapus Data Duplikat
customers_data = customers_data.drop_duplicates()

# Normalisasi atau Standarisasi Fitur
numerical_features = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']
scaler = StandardScaler()
customers_data[numerical_features] = scaler.fit_transform(customers_data[numerical_features])

# Encoding Data Kategorikal
categorical_features = ['Gender', 'Profession']
encoder = OneHotEncoder(sparse_output=False)
encoded_data = pd.DataFrame(encoder.fit_transform(customers_data[categorical_features]),
                            columns=encoder.get_feature_names_out(categorical_features))
customers_data = customers_data.drop(columns=categorical_features).reset_index(drop=True)
customers_data = pd.concat([customers_data, encoded_data], axis=1)

# Deteksi dan Penanganan Outlier (Menggunakan Z-Score)
z_scores = np.abs(customers_data[numerical_features].apply(lambda x: (x - x.mean()) / x.std()))
outliers = z_scores > 3  # Deteksi outlier di luar 3 standard deviation
customers_data = customers_data[~(outliers.any(axis=1))]  # Hapus outlier

# Menampilkan hasil preprocessing
customers_data.reset_index(drop=True, inplace=True)
print("\nPreprocessed Data Information:")
print(customers_data.info())

# Reduksi Dimensi dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(customers_data[numerical_features])

# Menentukan jumlah cluster yang optimal menggunakan metode Elbow (KMeans)
inertia = []
kmeans_silhouette_scores = []
range_clusters = range(2, 11)  # Jumlah cluster dari 2 hingga 10

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=500, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    inertia.append(kmeans.inertia_)
    kmeans_silhouette_scores.append(silhouette_score(X_pca, labels))

# Visualisasi Metode Elbow
plt.figure(figsize=(10, 6))
plt.plot(range_clusters, inertia, marker='o', linestyle='--', label='KMeans')
plt.title("Metode Elbow")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.legend()
plt.show()

# Visualisasi Silhouette Score (KMeans)
plt.figure(figsize=(10, 6))
plt.plot(range_clusters, kmeans_silhouette_scores, marker='o', linestyle='--', color='green', label='KMeans')
plt.title("Silhouette Score untuk KMeans")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)
dbscan_silhouette = silhouette_score(X_pca, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

# Agglomerative Clustering
agglo_silhouette_scores = []
for k in range_clusters:
    agglo = AgglomerativeClustering(n_clusters=k)
    labels = agglo.fit_predict(X_pca)
    agglo_silhouette_scores.append(silhouette_score(X_pca, labels))

# Visualisasi Silhouette Scores untuk Semua Algoritma
plt.figure(figsize=(12, 8))
plt.plot(range_clusters, kmeans_silhouette_scores, marker='o', label='KMeans', linestyle='--')
plt.plot(range_clusters, agglo_silhouette_scores, marker='s', label='Agglomerative', linestyle='--')
plt.axhline(y=dbscan_silhouette, color='r', linestyle='-', label='DBSCAN')
plt.title("Silhouette Scores untuk Berbagai Algoritma")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()

# Final Clustering dengan KMeans
optimal_clusters = 4  # Pilih jumlah cluster terbaik dari evaluasi
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=20, max_iter=500, random_state=42)
customers_data['Cluster'] = kmeans.fit_predict(X_pca)

# Visualisasi Hasil Clustering
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customers_data['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title(f"KMeans Clustering dengan {optimal_clusters} Cluster")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
