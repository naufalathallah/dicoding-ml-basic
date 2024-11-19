
"""# **2. Import Library**

Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning.
"""

#Type your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

"""# **3. Memuat Dataset**

Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.

Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.
"""

#Type your code here
pd.options.display.max_columns = None
file_path = 'Customers.csv'
df = pd.read_csv(file_path)

# Lihat informasi awal dataset
print(df.head())
print(df.info())

"""# **4. Exploratory Data Analysis (EDA)**

Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset. EDA bertujuan untuk:

1. **Memahami Struktur Data**
   - Tinjau jumlah baris dan kolom dalam dataset.  
   - Tinjau jenis data di setiap kolom (numerikal atau kategorikal).

2. **Menangani Data yang Hilang**  
   - Identifikasi dan analisis data yang hilang (*missing values*). Tentukan langkah-langkah yang diperlukan untuk menangani data yang hilang, seperti pengisian atau penghapusan data tersebut.

3. **Analisis Distribusi dan Korelasi**  
   - Analisis distribusi variabel numerik dengan statistik deskriptif dan visualisasi seperti histogram atau boxplot.  
   - Periksa hubungan antara variabel menggunakan matriks korelasi atau scatter plot.

4. **Visualisasi Data**  
   - Buat visualisasi dasar seperti grafik distribusi dan diagram batang untuk variabel kategorikal.  
   - Gunakan heatmap atau pairplot untuk menganalisis korelasi antar variabel.

Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.
"""

#Type your code here
# Mengecek missing values
print(df.isnull().sum())

# Statistik deskriptif
print(df.describe())

# Visualisasi distribusi variabel numerikal
numerical_cols = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.show()

# Korelasi antar variabel numerikal
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""# **5. Data Preprocessing**

Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning. Data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.

Berikut adalah tahapan-tahapan yang perlu dilakukan, namun **tidak terbatas** pada:
1. Menghapus atau Menangani Data Kosong (Missing Values)
2. Menghapus Data Duplikat
3. Normalisasi atau Standarisasi Fitur
4. Deteksi dan Penanganan Outlier
5. Encoding Data Kategorikal
6. Binning (Pengelompokan Data)
"""

#Type your code here
# Menangani missing values
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])

# Encoding data kategorikal
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Profession'] = label_encoder.fit_transform(df['Profession'])

# Standarisasi kolom numerikal
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']])

# Konversi kembali ke DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=['Age', 'Annual Income', 'Spending Score', 'Work Experience', 'Family Size'])

selected_features = ['Annual Income ($)', 'Spending Score (1-100)']
scaled_selected_data = scaler.fit_transform(df[selected_features])
scaled_selected_df = pd.DataFrame(scaled_selected_data, columns=selected_features)

# Periksa outlier menggunakan boxplot
numerical_cols = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Menghapus outlier berdasarkan IQR
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    df = df[mask]
    scaled_selected_df = scaled_selected_df[mask]  # Terapkan mask yang sama ke scaled_df

# Reset indeks pada kedua dataset agar sinkron
df = df.reset_index(drop=True)
scaled_selected_df = scaled_selected_df.reset_index(drop=True)

"""# **6. Pembangunan Model Clustering**

## **a. Pembangunan Model Clustering**

Pada tahap ini, Anda membangun model clustering dengan memilih algoritma yang sesuai untuk mengelompokkan data berdasarkan kesamaan. Berikut adalah **rekomendasi** tahapannya.
1. Pilih algoritma clustering yang sesuai.
2. Latih model dengan data menggunakan algoritma tersebut.
"""

#Type your code here
from sklearn.model_selection import ParameterGrid

# Mencari parameter optimal untuk KMeans
param_grid = {
    'n_clusters': range(2, 16),  # Range jumlah klaster
    'init': ['k-means++', 'random'],  # Metode inisialisasi
    'n_init': [10, 20],  # Iterasi inisialisasi
    'random_state': [42]
}

best_score = -1
best_params = None
for params in ParameterGrid(param_grid):
    kmeans = KMeans(**params)
    kmeans.fit(scaled_selected_df)  # Gunakan scaled_selected_df jika feature selection diterapkan
    score = silhouette_score(scaled_selected_df, kmeans.labels_)
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best Silhouette Score: {best_score} with params: {best_params}")

# Inisialisasi KMeans dengan parameter terbaik
kmeans = KMeans(**best_params)
kmeans.fit(scaled_selected_df)
df['Cluster'] = kmeans.labels_

# Pilih jumlah klaster terbaik berdasarkan Silhouette Score
optimal_k = 14  # Silhouette tertinggi
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(scaled_selected_df)

# Tambahkan label klaster ke dataset asli
df['Cluster'] = kmeans.labels_

# Kurangi dimensi menjadi 2 untuk PCA
pca = PCA(n_components=2)
scaled_df_pca = pca.fit_transform(scaled_selected_df)

# Clustering dengan data hasil PCA
kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_pca.fit(scaled_df_pca)
silhouette_pca = silhouette_score(scaled_df_pca, kmeans_pca.labels_)
print(f"Silhouette Score setelah PCA: {silhouette_pca}")

"""## **b. Evaluasi Model Clustering**

Untuk menentukan jumlah cluster yang optimal dalam model clustering, Anda dapat menggunakan metode Elbow atau Silhouette Score.

Metode ini membantu kita menemukan jumlah cluster yang memberikan pemisahan terbaik antar kelompok data, sehingga model yang dibangun dapat lebih efektif. Berikut adalah **rekomendasi** tahapannya.
1. Gunakan Silhouette Score dan Elbow Method untuk menentukan jumlah cluster optimal.
2. Hitung Silhouette Score sebagai ukuran kualitas cluster.
"""

#Type your code here

"""## **c. Feature Selection (Opsional)**

Silakan lakukan feature selection jika Anda membutuhkan optimasi model clustering. Jika Anda menerapkan proses ini, silakan lakukan pemodelan dan evaluasi kembali menggunakan kolom-kolom hasil feature selection. Terakhir, bandingkan hasil performa model sebelum dan sesudah menerapkan feature selection.
"""

#Type your code here

"""## **d. Visualisasi Hasil Clustering**

Setelah model clustering dilatih dan jumlah cluster optimal ditentukan, langkah selanjutnya adalah menampilkan hasil clustering melalui visualisasi.

Berikut adalah **rekomendasi** tahapannya.
1. Tampilkan hasil clustering dalam bentuk visualisasi, seperti grafik scatter plot atau 2D PCA projection.
"""

#Type your code here
plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_df_pca[:, 0], y=scaled_df_pca[:, 1], hue=kmeans_pca.labels_, palette='viridis', s=50)
plt.title('Clustering Results with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

"""## **e. Analisis dan Interpretasi Hasil Cluster**

Setelah melakukan clustering, langkah selanjutnya adalah menganalisis karakteristik dari masing-masing cluster berdasarkan fitur yang tersedia.

Berikut adalah **rekomendasi** tahapannya.
1. Analisis karakteristik tiap cluster berdasarkan fitur yang tersedia (misalnya, distribusi nilai dalam cluster).
2. Berikan interpretasi: Apakah hasil clustering sesuai dengan ekspektasi dan logika bisnis? Apakah ada pola tertentu yang bisa dimanfaatkan?
"""

#Type your code here
# Analisis karakteristik setiap cluster
for cluster in range(optimal_k):
    print(f'Cluster {cluster}:')
    cluster_data = df[df['Cluster'] == cluster]
    print(cluster_data.describe())
    print()

"""Tulis hasil interpretasinya di sini.
1. Cluster 1: Pelanggan dengan pendapatan rendah hingga menengah dan pengeluaran tinggi.
2. Cluster 2: Pelanggan dengan pendapatan tinggi tetapi pengeluaran rendah.
3. Cluster 3: Pelanggan dengan pendapatan menengah dan pengeluaran moderat.

# **7. Mengeksport Data**

Simpan hasilnya ke dalam file CSV.
"""

df.to_csv('clustered_data.csv', index=False)
print("Hasil clustering telah dieksport ke 'clustered_data.csv'")