import numpy as np


class kfConfig:
    def __init__(self, dyn_func, Q, R):
        self.dyn_func = dyn_func
        self.Q        = Q
        self.R        = R

class kfState:
    def __init__(self, x_kgk, P_kgk):
        self.x_kgk = x_kgk
        self.P_kgk = P_kgk

# def kf_predict(dyn_func, Q, x_kgk, P_kgk, u_k):

#     return x_kp1gk, P_kp1gk


# def kf_update
#     x_kp1gkm, P_kp1gk = kf_predict(kf_config.dyn_func, kf_config.Q, kf_state.x_kgk, kf_state.P_kgk, u_k)
    


# def kf(kf_config:kfConfig,kf_state:kfState, u_k, z_kp1):
#     x_kgkp1 = kf_config.dyn_func(kf_state.x_kgk, u_k)




