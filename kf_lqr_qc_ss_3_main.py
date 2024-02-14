import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.signal import dbode, dstep
from control import dlqr
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
    
    # create road profile and test window
    road_profile = road.CombinedRoadProfile()
    road_profile.add_new_feature(road.SingleRoadFeature(1.0, 0.05, 1.0, profile='halfsine'))
    road_profile.add_new_feature(road.SingleRoadFeature(4.0, -0.05, 1.0, profile='halfsine'))
    road_profile.add_new_feature(road.SingleRoadFeature(7.0, 0.025, 0.1, profile='rectangle'))  

    x_vel         = 5.0
    test_distance = 9.0
    sample_rate   = 500
    x_init_vec    = np.array([[0],[0],[0],[0]])
    u_vec         = np.array([[0]])

    # calculate time variables and vectors
    time_end        = test_distance/x_vel
    dt              = 1/sample_rate
    sample_time_vec = np.linspace(0,test_distance/x_vel, int(time_end/dt)+1)
    sample_x_vec    = sample_time_vec * x_vel

    # create discrete state space model
    qc_control_model, qc_disturb_model  = dyn.qc_ss_3(params, isdiscrete=True, ts = dt)
 
    # initialize kalman filter object
    x_dim, u_dim, y_dim = dyn.get_ss_dims(qc_control_model)
    _, ud_dim, _        = dyn.get_ss_dims(qc_disturb_model)

    kf   = KalmanFilter(x_dim, y_dim, u_dim)
    kf.x = x_init_vec
    kf.F = qc_control_model.A
    kf.H = qc_control_model.C
    kf.B = qc_control_model.B
    kf.Q = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])*1.0
    
    kf.R = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])*1.0
    
    # Create LQR controller instance
    Q_lqr = np.array([[1,0,0,0],
                      [0,5000,0,0],
                      [0,0,0.01,0],
                      [0,0,0,1000]])
    R_lqr = np.array([[1]])*0.0001
    k_lqr,_,_ = dlqr(qc_control_model.A, qc_control_model.B, Q_lqr, R_lqr)

    # prep log vectors
    x_gt_log         = np.zeros((len(sample_x_vec),x_dim,1))
    u_vec_log        = np.zeros((len(sample_x_vec),u_dim,1))   
    ud_vec_log       = np.zeros((len(sample_x_vec),ud_dim,1)) 
    x_est_log        = np.zeros((len(sample_x_vec),x_dim,1))
    z_log            = np.zeros((len(sample_x_vec),y_dim,1))
    z_prior_pred_log = np.zeros((len(sample_x_vec),y_dim,1))  

    #initialize simulation loop
    x_gt_log[0]    = x_init_vec
    x_est_log[0]   = x_init_vec
    u_vec_log[0]   = np.array([[0.0]])
    ud_vec_log[0]  = road_profile.get_z_pos(sample_x_vec[0]) 

    for idx, x_pos in enumerate(sample_x_vec):
        if idx == len(sample_x_vec)-1:
            break
        kf.predict(u=u_vec_log[idx])
        z_prior_pred_log[idx+1] = qc_control_model.C @ kf.x_prior + qc_control_model.D @ u_vec_log[idx] 
        # create process noise vector for proces contamination
        process_noise_array = (np.array([np.random.normal(0.0, 0.01, 4)])).transpose() 
        process_noise_array[0] = 0.0
        process_noise_array[2] = 0.0                                     
        # propagate state and grab new control/disturbance information at new state for direct feedthrough measurement 
        x_gt_log[idx+1]   = (qc_control_model.A @ x_gt_log[idx]) + (qc_control_model.B @ u_vec_log[idx]) + (qc_disturb_model.B @ ud_vec_log[idx]) +  process_noise_array                    
        ud_vec_log[idx+1] = road_profile.get_z_pos(sample_x_vec[idx+1]) 
        # create measurement noise vector for measurement contamination, take reading at state kp1   
        measure_noise_array = (np.array([np.random.normal(0.0, 0.1, 3)])).transpose()
        measure_noise_array[2] *= 0.1        
        z_log[idx+1] = (qc_control_model.C @ x_gt_log[idx+1]) + (qc_control_model.D @ u_vec_log[idx]) + (qc_disturb_model.D @ ud_vec_log[idx+1]) + measure_noise_array
        #calculate new kalman estimate of state
        kf.update(z_log[idx+1])
        x_est_log[idx+1] = kf.x
        u_vec_log[idx+1]  = -k_lqr @ x_est_log[idx+1]
        # z_est_log[idx+1] = qc_control_model.C @ kf.x + qc_control_model.D @ u_vec_log[idx]


    plt.figure(figsize=[18,10])
    plt.subplot(3,2,1)  
    plt.plot(sample_x_vec, ud_vec_log[:,0], marker='.', label='road z profile')  # 'o' creates circular markers at each data point 
    plt.plot(sample_x_vec, x_gt_log[:,0], marker='o',   label='sprung mass z')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_est_log[:,0], marker='D',  label='sprung mass z est')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_gt_log[:,2], marker='+',   label='unsprung mass z')  # 'o' creates circular markers at each data point       
    plt.plot(sample_x_vec, x_est_log[:,2], marker='x',  label='unsprung mass z est')  # 'o' creates circular markers at each data point  
    plt.title('Verical positions vs x position')
    plt.xlabel('position [m]')
    plt.ylabel('z [m]')
    plt.legend()    
    plt.grid(True)

    plt.subplot(3,2,2)  
    plt.plot(sample_x_vec, x_gt_log[:,1], marker='o',  label='sprung mass z vel')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_est_log[:,1], marker='D', label='sprung mass z vel est')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_gt_log[:,3], marker='+',  label='unsprung mass z vel')  # 'o' creates circular markers at each data point       
    plt.plot(sample_x_vec, x_est_log[:,3], marker='x', label='unsprung mass z vel est')  # 'o' creates circular markers at each data point  
    plt.title('Vertical velocities vs x position')
    plt.xlabel('position [m]')
    plt.ylabel('velocity [m/s]')
    plt.legend()    
    plt.grid(True)

    plt.subplot(3,2,3)
    plt.plot(sample_x_vec, z_log[:,0], marker='D',     label='sprung mass acc')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, z_prior_pred_log[:,0], marker='x', label='sprung mass acc est')  # 'o' creates circular markers at each data point       
    plt.title('Sprung Accekeration: GT vs estimate')
    plt.xlabel('position')
    plt.ylabel('zddot [m/s2]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,2,4)
    plt.plot(sample_x_vec, z_log[:,1], marker='D',     label='unsprung mass acc')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, z_prior_pred_log[:,1], marker='x', label='unsprung mass acc est')  # 'o' creates circular markers at each data point       
    plt.title('Unsprung Acceleration: GT vs estimate')
    plt.xlabel('position')
    plt.ylabel('zddot [m/s2]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,2,5)
    plt.plot(sample_x_vec, z_log[:,2], marker='D',     label='susp pot pos')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, z_prior_pred_log[:,2], marker='x', label='susp pot pos est')  # 'o' creates circular markers at each data point       
    plt.title('Suspension displacement: GT vs estimate')
    plt.xlabel('position')
    plt.ylabel('displacement [m]')
    plt.legend()   
    plt.grid(True)

    plt.subplot(3,2,6)
    plt.plot(sample_x_vec, u_vec_log[:,0], marker='D',     label='active susp force')  # 'o' creates circular markers at each data point   
    plt.title('Active suspension force vs position')
    plt.xlabel('position')
    plt.ylabel('Force [N]')
    plt.legend()   
    plt.grid(True)

    plt.tight_layout()
    plt.show()