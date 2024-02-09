import numpy as np
from scipy.signal import StateSpace as ss

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




