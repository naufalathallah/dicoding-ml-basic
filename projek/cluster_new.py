import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

file_path = 'Customers.csv'
customers_data = pd.read_csv(file_path)

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
customers_data.head(), customers_data.info()

# Menentukan jumlah cluster yang optimal menggunakan metode Elbow
X = customers_data[numerical_features]

inertia = []
silhouette_scores = []
range_clusters = range(2, 11)  # Jumlah cluster dari 2 hingga 10

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Visualisasi Metode Elbow
plt.figure(figsize=(10, 6))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title("Metode Elbow")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.show()

# Visualisasi Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(range_clusters, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title("Silhouette Score untuk Berbagai Cluster")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.show()

# Melatih model clustering dengan jumlah cluster optimal (misalnya, 4 berdasarkan hasil)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customers_data['Cluster'] = kmeans.fit_predict(X)

# Visualisasi Hasil Clustering dalam 2D (menggunakan PCA untuk reduksi dimensi jika perlu)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customers_data['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title("Visualisasi Hasil Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()