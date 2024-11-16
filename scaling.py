from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Contoh data
data = [[10], [2], [30], [40], [50]]

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
scaled_min_max = min_max_scaler.fit_transform(data)
print("Min-Max Scaling:\n", scaled_min_max)

# Standardization
standard_scaler = StandardScaler()
scaled_standard = standard_scaler.fit_transform(data)
print("\nStandardization:\n", scaled_standard)