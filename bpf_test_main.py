import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt
import src.dsp_funcs as dsp

fs = 1000 
t = np.arange(0, 1, 1/fs)
f1 = 10  
f2 = 50  
pure_sig = np.sin(2*np.pi*(f2+f1)/2*t)

noise_std = 1.0
noise = np.random.normal(1.0, noise_std, pure_sig.shape)
noisy_sig = pure_sig + noise  # The signal with added white noise

# Continue
order = 4  # Filter order
sos = butter(order, [f1, f2], fs=fs, btype='bandpass', analog=False, output='sos')

# Step 2: Initialize the filter state
z = sosfilt_zi(sos)
filtered_sig = []
# Assuming `incoming_data_stream` is an iterable with your incoming data points
for new_data_point in noisy_sig:
    # Step 3: Apply the filter as data arrives
    filtered_data_point, z = sosfilt(sos, [new_data_point], zi=z)
    filtered_sig.append(filtered_data_point[0])  # Store the filtered data

filtered_sig = np.array(filtered_sig)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, noisy_sig, label='Original Signal')
plt.plot(t, filtered_sig, label='Filtered Signal', linestyle='--')
plt.title('Signal Before and After Bandpass Filtering')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()