import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset hasil clustering
file_path = 'clustered_data.csv'
data = pd.read_csv(file_path)

# Menampilkan beberapa baris data untuk verifikasi
print(data.head())

# Memisahkan fitur (X) dan target (y)
X = data.drop(columns=['Cluster'])  # Cluster adalah target
y = data['Cluster']  # Target cluster

# Standardisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Data splitting: 80% untuk latih, 20% untuk uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Menampilkan ukuran data yang di-split
print(f"Ukuran data latih: {X_train.shape}, data uji: {X_test.shape}")

# Melatih beberapa model klasifikasi
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"Model {name} telah dilatih.")

# Evaluasi setiap model
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\nEvaluasi untuk model: {name}")
    print(f"\nCross-Validation Accuracy untuk {name}: {scores.mean():.2f} (+/- {scores.std():.2f})")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

# Confusion Matrix untuk melihat detail prediksi benar dan salah
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

"""
### **Analisis Hasil Evaluasi Model Klasifikasi**

#### **1. Identifikasi Kelemahan Model**
- **Precision atau Recall Rendah untuk Kelas Tertentu:**
  - Berdasarkan **Confusion Matrix**, terlihat bahwa beberapa model seperti **Logistic Regression** dan **KNN** memiliki **kesalahan klasifikasi minor**, terutama untuk kelas dengan representasi lebih kecil, seperti **Cluster 1, 2, dan 3**.
  - **KNN** menunjukkan kesalahan klasifikasi yang lebih tinggi pada kelas minoritas, khususnya Cluster 1 dan Cluster 2, dibandingkan model lainnya.

- **Potensi Overfitting:**
  - **Decision Tree** dan **Random Forest** menunjukkan **kinerja sempurna (Accuracy = 1.00)** pada data uji.
  - Hasil ini menunjukkan kemungkinan **overfitting**, terutama pada Decision Tree, yang cenderung mempelajari data terlalu detail tanpa generalisasi yang baik.

- **Distribusi Kelas Tidak Seimbang:**
  - **Cluster 0** memiliki jumlah data yang jauh lebih besar dibandingkan kelas lainnya, yang dapat memengaruhi kemampuan model untuk mengenali kelas minoritas seperti **Cluster 2** dan **Cluster 3**.

#### **2. Rekomendasi Tindakan Lanjutan**
1. **Pemeriksaan Potensi Overfitting:**
   - Gunakan dataset baru (eksternal) untuk memvalidasi performa model guna memastikan generalisasi yang baik.
   - Pada **Decision Tree**, tambahkan parameter seperti `max_depth` atau `min_samples_split` untuk membatasi kompleksitas model.

2. **Hyperparameter Tuning:**
   - Lakukan tuning untuk model Random Forest, Logistic Regression, dan KNN untuk meningkatkan akurasi dan generalisasi:
     - **Random Forest:**
       - Coba parameter seperti `n_estimators`, `max_depth`, dan `min_samples_split`.
     - **Logistic Regression:**
       - Eksperimen dengan solver seperti `saga` atau `newton-cg` untuk stabilitas konvergensi.
     - **KNN:**
       - Uji berbagai nilai `n_neighbors` dan metrik jarak seperti `manhattan` atau `minkowski`.

3. **Penanganan Kelas Tidak Seimbang:**
   - Terapkan teknik **oversampling** seperti **SMOTE (Synthetic Minority Over-sampling Technique)** untuk memperkuat representasi kelas minoritas.
   - Alternatifnya, gunakan **class weights** pada model Logistic Regression atau Random Forest untuk mengatasi ketidakseimbangan kelas.

4. **Pengumpulan Data Tambahan:**
   - Jika memungkinkan, tambahkan lebih banyak data untuk kelas minoritas (Cluster 1, 2, dan 3) untuk meningkatkan kemampuan model mendeteksi cluster ini.

5. **Interpretasi dan Penggunaan Hasil:**
   - Gunakan hasil klasifikasi untuk mendukung strategi bisnis, seperti:
     - Meningkatkan program loyalitas untuk pelanggan di Cluster 0.
     - Memberikan promosi eksklusif untuk pelanggan di Cluster 2 dan 3.
     - Mengembangkan strategi pemasaran yang lebih personal berdasarkan cluster.

#### **Kesimpulan**
- Semua model menunjukkan kinerja tinggi dengan **Accuracy**, **Precision**, **Recall**, dan **F1-Score** yang sangat baik.
- **Decision Tree** dan **Random Forest** perlu diperiksa lebih lanjut untuk kemungkinan overfitting.
- **Logistic Regression** dan **KNN** memerlukan penyesuaian tambahan untuk meningkatkan performa pada kelas minoritas.
- Penyesuaian tambahan melalui hyperparameter tuning, penanganan data tidak seimbang, atau algoritma baru dapat digunakan untuk meningkatkan hasil lebih lanjut.
"""