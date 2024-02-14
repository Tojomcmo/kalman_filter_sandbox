import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.signal import bode, step, dbode, dstep
import src.dyn_funcs as dyn
import src.road_profile_funcs as road
import matplotlib.pyplot as plt

if __name__ == "__main__":

    params = {
              'ms'  : 400,   
              'mu'  : 50,
              'bs'  : 1000,
              'ks'  : 35000,
              'kt'  : 190000,
            }
   
    dt = 1/500
    output_select = 2
    output_select_list = ['sprung acceleration', 
                          'unsprung acceleration',
                          'suspension displacement']

    qc_control_model, qc_disturb_model  = dyn.qc_ss_3(params, isdiscrete=False, ts = dt)

    qc_siso = qc_disturb_model
    qc_siso.C = qc_siso.C[output_select,:]
    #qc_siso.C = np.array([[0,0,1,0]])
    qc_siso.D = qc_siso.D[output_select,:]


    # Set frequency bounds (in Hz) and convert to rad/s
    f_min = 0.1  # Minimum frequency in Hz
    f_max = 100  # Maximum frequency in Hz
    w_min = 2 * np.pi * f_min  # Convert to rad/s
    w_max = 2 * np.pi * f_max  # Convert to rad/s
    # Generate frequency range (in rad/s)
    w = np.logspace(np.log10(w_min), np.log10(w_max), 5000)
    f = w / (2 * np.pi)
    w, mag, phase = bode(qc_siso, w=w)    
    
    t_vec = np.linspace(0,5,5001)
    t, y = step(qc_siso,T=t_vec)

    plt.figure(figsize=[12,12])
    plt.suptitle(f'SISO output selection: {output_select_list[output_select]}')
    plt.subplot(3,1,1)
    plt.semilogx(f, mag)
    plt.title('Bode Plot - Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True, which="both")

    plt.subplot(3,1,2)
    plt.semilogx(w, phase)
    plt.title('Bode Plot - Phase')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [deg]')
    plt.grid(True, which="both")


    plt.subplot(3,1,3)
    plt.plot(t, y)
    plt.title('Step Response')
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.grid(True)
    plt.tight_layout()
    plt.show()