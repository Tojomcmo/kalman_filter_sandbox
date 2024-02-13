import numpy as np
from scipy.signal import StateSpace as ss
from scipy.signal import butter, filtfilt

def dynamics_rk4_zoh(dyn_func, ts, x, u):
    '''
    This function calculates fourth order Runge Kutta integration of a dynamics function
    of state vector x and control vector u. A zero order hold is placed on u, and the integration
    calculates the new state vector at the provided sample time ts.
    
    dyn_func - [in]: dynamics function that accepts x and u as inputs and returns x_dot
    ts       - [in]: time step for single step integration
    x        - [in]: state vector compatible with dynamics function
    u        - [in]: control vector compatible with dynamics function
    return   - [out]: (x + dx) where dx is calculated by dyn_func integration over ts time
    '''
    f1 = dyn_func(x, u)
    f2 = dyn_func(x + (ts/2)*f1, u)
    f3 = dyn_func(x + (ts/2)*f2, u)
    f4 = dyn_func(x + ts*f3, u)
    return x + ((ts/6) * (f1 + (f2/2) + (f3/2) + f4))




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