import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.model_selection import train_test_split
import joblib
import pickle

pd.options.display.max_columns = None

test = pd.read_csv("content/test.csv")
print(test.head())

train = pd.read_csv("content/train.csv")
print(train.head())

train.info()
print(train.describe(include="all"))

missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])

less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

print(less)
print(over)

numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

kategorical_features = train[less].select_dtypes(include=['object']).columns

for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

df = train.drop(columns=over)

missing_values = df.isnull().sum()
print("missing lebih dari 1000: ", missing_values[missing_values > 0])

# for feature in numeric_features:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box Plot of {feature}')
    # plt.show()

Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_filtered_numeric = df.loc[condition, numeric_features]

# Menggabungkan kembali dengan kolom kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)

# for feature in numeric_features:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# sns.histplot(train[numeric_features[3]], kde=True)
# plt.title("Histogram Sebelum Standardisasi")
# plt.show()

# plt.subplot(1, 2, 2)
# sns.histplot(df[numeric_features[3]], kde=True)
# plt.title("Histogram Setelah Standardisasi")
# plt.show()

duplicates = df.duplicated()

print("Baris duplikat:")
print(df[duplicates])

df = df.drop_duplicates()

print("DataFrame setelah menghapus duplikat:")
print(df)

category_features = df.select_dtypes(include=['object']).columns
print(df[category_features])

df_one_hot = pd.get_dummies(df, columns=category_features)
print(df_one_hot)

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

X = df_lencoder.drop(columns=['SalePrice'])
y = df_lencoder['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# menghitung panjang/jumlah data
print("Jumlah data: ",len(X))
# menghitung panjang/jumlah data pada x_test
print("Jumlah data latih: ",len(x_train))
# menghitung panjang/jumlah data pada x_test
print("Jumlah data test: ",len(x_test))

# Melatih model 1 dengan algoritma Least Angle Regression
from sklearn import linear_model

lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)

# Melatih model 2 dengan algoritma Linear Regression
from sklearn.linear_model import LinearRegression

LR = LinearRegression().fit(x_train, y_train)

# Melatih model 3 dengan algoritma Gradient Boosting Regressor
GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(x_train, y_train)

# Evaluasi pada model LARS
pred_lars = lars.predict(x_test)
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)

# Membuat dictionary untuk menyimpan hasil evaluasi
data = {
    'MAE': [mae_lars],
    'MSE': [mse_lars],
    'R2': [r2_lars]
}

# Konversi dictionary menjadi DataFrame
df_results = pd.DataFrame(data, index=['Lars'])
print(df_results)

# Evaluasi pada model Linear Regression
pred_LR = LR.predict(x_test)
mae_LR = mean_absolute_error(y_test, pred_LR)
mse_LR = mean_squared_error(y_test, pred_LR)
r2_LR = r2_score(y_test, pred_LR)

# Menambahkan hasil evaluasi LR ke DataFrame
df_results.loc['Linear Regression'] = [mae_LR, mse_LR, r2_LR]
print(df_results)

# Evaluasi pada model Linear Regression
pred_GBR = GBR.predict(x_test)
mae_GBR = mean_absolute_error(y_test, pred_GBR)
mse_GBR = mean_squared_error(y_test, pred_GBR)
r2_GBR = r2_score(y_test, pred_GBR)

# Menambahkan hasil evaluasi LR ke DataFrame
df_results.loc['GradientBoostingRegressor'] = [mae_GBR, mse_GBR, r2_GBR]
print(df_results)

# Menyimpan model ke dalam file
joblib.dump(GBR, 'gbr_model.joblib')

# Menyimpan model ke dalam file
with open('gbr_model.pkl', 'wb') as file:
    pickle.dump(GBR, file)