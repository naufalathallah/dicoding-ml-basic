from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import time

# Mengunduh dataset California Housing
X, y = fetch_california_housing(return_X_y=True)

# Membagi dataset menjadi training set dan testing set (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melakukan scaling pada data (penting untuk regresi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Shape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)

# Inisialisasi model Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Evaluasi awal model tanpa tuning
y_pred = rf.predict(X_test)
initial_mse = mean_squared_error(y_test, y_pred)
print(f"Initial MSE on test set (without tuning): {initial_mse:.2f}")

start_time = time.time()  # Mencatat waktu mulai

# Definisikan parameter grid untuk Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output hasil terbaik
print(f"Best parameters (Grid Search): {grid_search.best_params_}")
best_rf_grid = grid_search.best_estimator_

# Evaluasi performa model setelah Grid Search
y_pred_grid = best_rf_grid.predict(X_test)
grid_search_mse = mean_squared_error(y_test, y_pred_grid)
print(f"MSE after Grid Search: {grid_search_mse:.2f}")
end_time = time.time()  # Mencatat waktu selesai
execution_time = end_time - start_time  # Menghitung selisih waktu
print(f"Waktu eksekusi: {execution_time:.4f} detik")