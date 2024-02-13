import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.signal import butter, sosfilt, sosfilt_zi

import src.dyn_funcs as dyn
import src.road_profile_funcs as road

if __name__ == "__main__":

    params = {
              'ms'  : 300,   
              'mu'  : 50,
              'bs'  : 1000,
              'ks'  : 35000,
              'kt'  : 190000,
            }
    
    # create road profile and test window
    road_profile  = road.SingleRoadFeature(1.0, 0.05, 1.0, profile='halfsine')    
    x_vel         = 2.0
    test_distance = 8.0
    sample_rate   = 200
    x_init_vec    = np.array([[0],[0],[0],[0],[0]])
    u_vec         = np.array([[0],[0]])

    # calculate time variables and vectors
    time_end        = test_distance/x_vel
    dt              = 1/sample_rate
    sample_time_vec = np.linspace(0,test_distance/x_vel, int(time_end/dt)+1)
    sample_x_vec    = sample_time_vec * x_vel

    # create discrete state space model
    qc_d_model  = dyn.qc_ss_1(params, isdiscrete=True, ts = dt)

    # initialize kalman filter object
    x_dim, u_dim, y_dim = dyn.get_ss_dims(qc_d_model)
    kf   = KalmanFilter(x_dim, y_dim, u_dim)
    kf.x = x_init_vec
    kf.F = qc_d_model.A
    kf.H = qc_d_model.C
    kf.B = qc_d_model.B
    kf.Q = np.array([[1,0,0,0,0],
                     [0,1,0,0,0],
                     [0,0,1,0,0],
                     [0,0,0,1,0],
                     [0,0,0,0,1]])*0.1
    kf.R = np.array([[1,0,0],
                     [0,1,0],
                     [0,0,1]])

    #define bandpass filter for velocity estimation
    f1 = 10  
    f2 = 180 
    order = 2  # Filter order
    #sos = butter(order, [f1, f2], fs=sample_rate, btype='bandpass', analog=False, output='sos')
    sos = butter(order, [f1], fs=sample_rate, btype='highpass', analog=False, output='sos')    
    z = sosfilt_zi(sos) * 0
    vel_filt_log = []

    # prep log vectors
    u_vec_log    = np.zeros((len(sample_x_vec),u_dim,1))    
    x_gt_log     = np.zeros((len(sample_x_vec),x_dim,1))
    x_gt_log[0]  = x_init_vec
    x_est_log    = np.zeros((len(sample_x_vec),x_dim,1))
    x_est_log[0] = x_init_vec
    vel_filt_log = []
    vel_est_prev = 0.0


    u_est = np.array([[0.0],[0.0]])
    for idx, x_pos in enumerate(sample_x_vec):
        if idx == len(sample_x_vec)-1:
            break
        # update input vector with current road velocity 
        u_vec_log[idx] =np.array([[0],
                                  [road_profile.get_z_vel(x_pos, x_vel)]])                                      
        # propagate state and measurement
        x_gt_log[idx+1] = np.matmul(qc_d_model.A,x_gt_log[idx]) + np.matmul(qc_d_model.B,u_vec_log[idx])
        noise_1 = np.random.normal(0.0, 0.001, 1)
        noise_2 = np.random.normal(0.0, 0.001, 1)
        noise_3 = np.random.normal(0.0, 0.001, 1) 
        noise_array = np.array([noise_1,noise_2,noise_3])      
        y_kp1 = np.matmul(qc_d_model.C,x_gt_log[idx+1]) 
        z_kp1 = y_kp1 + noise_array
        #calculate new kalman estimate
        if idx > 1:
            vel_calc  = (x_est_log[idx,4]-x_est_log[idx-1,4])/dt
            vel_filt, z = sosfilt(sos, vel_calc, zi=z)
            vel_filt_log.append(vel_filt)  # Store the filtered data
            u_est[1]  = vel_filt
            
        kf.predict(u=u_est)
        kf.update(z_kp1)
        x_est_log[idx+1] = kf.x

    vel_filt_log = np.array(vel_filt_log)
    # Create the plot   
    plt.plot(sample_x_vec, x_gt_log[:,0], marker='o', label='sprung mass z')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_gt_log[:,2], marker='+', label='unsprung mass z')  # 'o' creates circular markers at each data point       
    plt.plot(sample_x_vec, x_gt_log[:,4], marker='x', label='road z profile')  # 'o' creates circular markers at each data point    

    plt.plot(sample_x_vec, x_est_log[:,0], marker='o', label='sprung mass z est')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_est_log[:,2], marker='+', label='unsprung mass z est')  # 'o' creates circular markers at each data point       
    plt.plot(sample_x_vec, x_est_log[:,4], marker='x', label='road z profile est')  # 'o' creates circular markers at each data point  
    # Add title and labels
    plt.title('GT vertical pos vs x pos')
    plt.xlabel('position')
    plt.ylabel('z')
    plt.legend()

    # Optional: Add grid lines
    plt.grid(True)
    # Show the plot
    plt.show()