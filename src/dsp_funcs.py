import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   y = filtfilt(b, a, data)
   return y

fs = 1000 
t = np.arange(0, 1, 1/fs)
f1 = 10  
f2 = 50  
sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
order = 4  
filtered = butter_bandpass_filter(sig, f1, f2, fs, order)