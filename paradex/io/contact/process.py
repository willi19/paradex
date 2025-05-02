import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

# === FILTER FUNCTIONS ===
def median_filtering(data, window_size=5):
    return median_filter(data, size=(window_size, 1))

def savgol_filtering(data, window_size=51, polyorder=3):
    return savgol_filter(data, window_length=window_size, polyorder=polyorder, axis=0)

# === PROCESSING FUNCTION ===
def process_contact(data, method="median", trim=100, **kwargs):
    filtered = median_filtering(data, window_size=240)
    return filtered[:, :15]

# def moving_average(data, window_size=10):   
#     half_window = window_size // 2
#     # Create an array for filtered data
#     filtered_data = np.zeros_like(data)

#     # Apply moving average filter to each column (sensor)
#     for i in range(data.shape[1]):  # Loop over all 15 sensors
#         # Apply symmetric padding
#         padded_data = np.pad(data[:, i], (half_window, half_window), mode='symmetric')

#         # Apply the moving average filter
#         padded_filtered_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='same')

#         # Remove padding effects (first and last half_window values)
#         filtered_data[:, i] = padded_filtered_data[half_window:-half_window]

#     filtered_data = filtered_data/filtered_data[0,:] - 1
#     return filter_data

# # === Define Base Directory (Change this to your main folder) ===
# base_dir = "D:/Mocap/capture data/capture/bookstand/1/"

# # === Load Data ===
# data_file = os.path.join(base_dir, "contact/data.npy")
# timestamp_file = os.path.join(base_dir, "contact/timestamp.npy")

# data = np.load(data_file)
# timestam = np.load(timestamp_file)
# timestam = timestam - timestam[0]  # Normalize time
# timestam = timestam.flatten()

# # Define window size for moving average


# filter_data = moving_average(data)

# # Create DataFrame for saving the processed data
# columns = ["Timestamp"] + [f"Sensor_{i+1}_Noisy" for i in range(data.shape[1])] + [f"Sensor_{i+1}_Filtered" for i in range(data.shape[1])]
# df_data = np.column_stack((timestam, data, filtered_data))  # Combine timestamp, noisy data, and filtered data

# df = pd.DataFrame(df_data, columns=columns)

# # Save as CSV
# csv_filename = os.path.join(base_dir, "contact/data.csv")
# df.to_csv(csv_filename, index=False)  # Save without an extra index column

# print(f"Data saved to {csv_filename}")

# # === SAVE AS .NPY FILES ===
# filtered_data_npy_path = os.path.join(base_dir, "contact/data_filtered.npy")
# timestamp_npy_path = os.path.join(base_dir, "contact/timestamp_filtered.npy")

# np.save(filtered_data_npy_path, filtered_data)
# np.save(timestamp_npy_path, timestam)

# print(f"Filtered data saved as NPY: {filtered_data_npy_path}")
# print(f"Timestamp saved as NPY: {timestamp_npy_path}")

# # === PLOT 1: Noisy Data (Original) ===
# plt.figure(figsize=(12, 6))
# for i in range(data.shape[1]):
#     plt.plot(timestam, data[:, i], alpha=0.5, label=f"Sensor {i+1}")  # Faded for better visibility

# plt.xlabel("Time (s)")
# plt.ylabel("Sensor Value")
# plt.title("Noisy Sensor Data (Original)")
# plt.legend(loc="upper left", fontsize="small", ncol=3)
# plot_path = os.path.join(base_dir, "contact/noisy.png")
# plt.savefig(plot_path, dpi=500, bbox_inches='tight')
# plt.show()

# # === PLOT 2: Filtered Data (Moving Average Applied) ===
# plt.figure(figsize=(12, 6))
# for i in range(filtered_data.shape[1]):
#     plt.plot(timestam, filtered_data[:, i], label=f"Sensor {i+1}")

# plt.xlabel("Time (s)")
# plt.ylabel("Sensor Value")
# plt.title("Filtered Sensor Data (Moving Average Applied)")
# plt.legend(loc="upper left", fontsize="small", ncol=3)
# plot_path = os.path.join(base_dir, "contact/filtered.png")
# plt.savefig(plot_path, dpi=500, bbox_inches='tight')
# plt.show()