import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
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

# Menambahkan Fitur Baru
customers_data['Spending per Age'] = customers_data['Spending Score (1-100)'] / (customers_data['Age'] + 1)
customers_data['Income per Experience'] = customers_data['Annual Income ($)'] / (customers_data['Work Experience'] + 1)

# Menampilkan hasil preprocessing
customers_data.reset_index(drop=True, inplace=True)
print("\nPreprocessed Data Information:")
print(customers_data.info())

# Reduksi Dimensi dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(customers_data[numerical_features + ['Spending per Age', 'Income per Experience']])

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

# Optimasi DBSCAN
param_grid = {'eps': [0.3, 0.5, 0.7], 'min_samples': [3, 5, 7]}
best_silhouette = -1
best_params = None
for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(X_pca)
    if len(set(labels)) > 1:
        score = silhouette_score(X_pca, labels)
        if score > best_silhouette:
            best_silhouette = score
            best_params = params
print(f"Best DBSCAN params: {best_params}, Silhouette Score: {best_silhouette}")

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
plt.axhline(y=best_silhouette, color='r', linestyle='-', label=f'DBSCAN: {best_silhouette:.2f}')
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
