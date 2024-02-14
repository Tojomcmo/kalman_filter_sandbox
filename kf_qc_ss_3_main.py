import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.signal import dbode, dstep
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
    road_profile.add_new_feature(road.SingleRoadFeature(7.0, 0.05, 2.0, profile='rectangle'))  

    x_vel         = 3.0
    test_distance = 12.0
    sample_rate   = 200
    x_init_vec    = np.array([[0],[0],[0],[0]])
    u_vec         = np.array([[0]])

    # calculate time variables and vectors
    time_end        = test_distance/x_vel
    dt              = 1/sample_rate
    sample_time_vec = np.linspace(0,test_distance/x_vel, int(time_end/dt)+1)
    sample_x_vec    = sample_time_vec * x_vel

    # create discrete state space model
    qc_control_model, qc_disturb_model  = dyn.qc_ss_3(params, isdiscrete=True, ts = dt)

    dbode(qc_disturb_model)
 
    # initialize kalman filter object
    x_dim, u_dim, y_dim = dyn.get_ss_dims(qc_control_model)
    _, ud_dim, _ = dyn.get_ss_dims(qc_disturb_model)

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
                     [0, 0, 1]])*0.1

    # prep log vectors
    x_gt_log     = np.zeros((len(sample_x_vec),x_dim,1))
    u_vec_log    = np.zeros((len(sample_x_vec),u_dim,1))   
    ud_vec_log   = np.zeros((len(sample_x_vec),ud_dim,1)) 
    x_est_log    = np.zeros((len(sample_x_vec),x_dim,1))
    x_gt_log[0]  = x_init_vec
    x_est_log[0] = x_init_vec
    z_log        = np.zeros((len(sample_x_vec),y_dim,1))
    z_est_log    = np.zeros((len(sample_x_vec),y_dim,1))    

    u_est = np.array([[0.0]])
    u_vec_log[0]    = u_est 
    ud_vec_log[0] = road_profile.get_z_pos(sample_x_vec[0]) 
    for idx, x_pos in enumerate(sample_x_vec):
        if idx == len(sample_x_vec)-1:
            break
        kf.predict(u=u_vec_log[idx])
        # grab predicted measurement of current state

        process_noise_array = (np.array([np.random.normal(0.0, 0.001, 4)])).transpose() 
        process_noise_array[0] = 0.0
        process_noise_array[2] = 0.0                                     
        # propagate state and take measurement
        x_gt_log[idx+1] = (qc_control_model.A @ x_gt_log[idx]) + (qc_control_model.B @ u_vec_log[idx]) + (qc_disturb_model.B @ ud_vec_log[idx]) +  process_noise_array                    
        u_vec_log[idx+1]  = np.array([[0.0]])  
        ud_vec_log[idx+1] = road_profile.get_z_pos(sample_x_vec[idx+1])

        z_kp1gk = qc_control_model.C @ kf.x_prior + qc_control_model.D @ u_vec_log[idx]
        z_est_log[idx+1] = z_kp1gk    
           
        measure_noise_array = (np.array([np.random.normal(0.0, 0.01, 3)])).transpose()
        measure_noise_array[2] *= 0.1        
        z_kp1   = (qc_control_model.C @ x_gt_log[idx+1]) + (qc_control_model.D @ u_vec_log[idx]) + (qc_disturb_model.D @ ud_vec_log[idx+1]) + measure_noise_array
        #calculate new kalman estimate of state
        kf.update(z_kp1)
        x_est_log[idx+1] = kf.x
        z_log[idx+1]     = z_kp1
        # z_est_log[idx+1] = qc_control_model.C @ kf.x + qc_control_model.D @ u_vec_log[idx]


    fig1 = plt.figure
    plt.subplot(2,2,1)  
    plt.plot(sample_x_vec, ud_vec_log[:,0], marker='.', label='road z profile')  # 'o' creates circular markers at each data point 
    plt.plot(sample_x_vec, x_gt_log[:,0], marker='o', label='sprung mass z')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_gt_log[:,2], marker='+', label='unsprung mass z')  # 'o' creates circular markers at each data point       
    plt.plot(sample_x_vec, x_est_log[:,0], marker='D', label='sprung mass z est')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, x_est_log[:,2], marker='x', label='unsprung mass z est')  # 'o' creates circular markers at each data point  
    plt.title('GT vertical pos vs x pos')
    plt.xlabel('position')
    plt.ylabel('z')
    plt.legend()    
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(sample_x_vec, z_log[:,0], marker='D', label='sprung mass acc')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, z_est_log[:,0], marker='x', label='sprung mass acc est')  # 'o' creates circular markers at each data point       
    plt.title('Sprung GT Acc vs Acc estimate')
    plt.xlabel('position')
    plt.ylabel('zddot')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(sample_x_vec, z_log[:,1], marker='D', label='unsprung mass acc')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, z_est_log[:,1], marker='x', label='unsprung mass acc est')  # 'o' creates circular markers at each data point       
    plt.title('Unsprung GT Acc vs Acc estimate')
    plt.xlabel('position')
    plt.ylabel('zddot')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(sample_x_vec, z_log[:,2], marker='D', label='susp pot pos')  # 'o' creates circular markers at each data point
    plt.plot(sample_x_vec, z_est_log[:,2], marker='x', label='susp pot pos est')  # 'o' creates circular markers at each data point       
    plt.title('GT susp disp vs susp disp estimate')
    plt.xlabel('position')
    plt.ylabel('displacement')
    plt.legend()   
    plt.grid(True)

    plt.tight_layout()
    plt.show()