import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


# Create a small array
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], dtype=float)

# Flatten the array to apply scaling on the entire dataset
data_flat = data.reshape(-1, 1)

# Initialize the MinMaxScaler and transform
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data_flat = scaler.fit_transform(data_flat)


scaler = StandardScaler()
scaled_data_flat = scaler.fit_transform(data_flat)  # Scaling applied to the whole array
scaled_data = scaled_data_flat.reshape(data.shape)  # Reshape back

# Reshape back to the original shape
scaled_data = scaled_data_flat.reshape(data.shape)

print("Original Data:\n", data)
print("Scaled Data (Whole Array):\n", scaled_data)


# Print mean and standard deviation after scaling
print("Mean after scaling (should be close to 0):\n", scaled_data.mean())
print("Std after scaling (should be close to 1):\n", scaled_data.std())

def standard_scale(data):
    means = np.mean(data, axis=(0, 1), keepdims=True)
    stds = np.std(data, axis=(0, 1), keepdims=True)
    return (data - means) / stds

def min_max_scale_per_channel(data, feature_range=(0, 1)):
    data_scaled = np.zeros_like(data)
    for c in range(data.shape[0]):  # Loop over channels
        channel = data[c]
        min_val = np.min(channel)
        max_val = np.max(channel)
        if max_val == min_val:
            data_scaled[c] = feature_range[0]
        else:
            scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
            data_scaled[c] = scale * (channel - min_val) + feature_range[0]
    return data_scaled



scaled_data2 = standard_scale(data)  # Reshape back

# Reshape back to the original shape
# scaled_data = scaled_data.reshape(data.shape)

print("Original Data:\n", data)
print("Scaled Data (Whole Array):\n", scaled_data2)
# Print mean and standard deviation after scaling
print("Mean after scaling (should be close to 0):\n", scaled_data2.mean())
print("Std after scaling (should be close to 1):\n", scaled_data2.std())
