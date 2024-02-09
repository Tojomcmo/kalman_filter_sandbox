import numpy as np
from scipy.signal import StateSpace, cont2discrete
from scipy.linalg import expm


def qc_ss_1(params, isdiscrete = False, ts = 0.0):
   '''
    This function generates the state space model for an active suspension Quarter Car model
    with the provided parameter dictionary
    
    params - [in]: dictionary with following structure
       'ms' - sprung mass [kg]
       'mu' - unsprung mass [kg]
       'bs' - suspension damping rate [Ns/m]
       'ks' - suspension spring rate [N/m]
       'kt' - tire spring rate [N/m]
    
    The state space structure corresponds to the following vectors   
    x: state vector [5x1] 
       zs      - sprung mass position [m]
       zs_dot  - sprung mass velocity [m/s]
       zu      - unsprung mass position [m]
       zu_dot  - unsprung mass velocity [m/s]
       zr      - road profile vertical position [N]
    u: input vector [2x1]
       fs      - active suspension force [N]
       zr_dot  - road profile vertical velocity [N]
    y: output vector [3x1]
       zs_ddot   - sprung mass acceleration [m/s2]  
       zu_ddot   - unsprung mass acceleration [m/s2]
       susp_disp - suspension displacement (zs-zu) [m]
    '''
   ms = params['ms']
   mu = params['mu']
   bs = params['bs']
   ks = params['ks']
   kt = params['kt']

   A = np.array([[0, 1, 0, 0, 0],
                  [-(ks/ms), -(bs/ms), (ks/ms), (bs/ms), 0],
                  [0, 0, 0, 1, 0],
                  [(ks/mu), (bs/mu), -((ks+kt)/mu), -(bs/mu), (kt/mu)],
                  [0, 0, 0, 0, 0]])
    
   B = np.array([[0, 0],
                  [(1/ms), 0],
                  [0, 0],
                  [-(1/mu), 0],
                  [0, 1]])
    
   C = np.array([[-(ks/ms), -(bs/ms), (ks/ms), (bs/ms), 0],
                  [(ks/mu), (bs/mu), -((ks+kt)/mu), -(bs/mu), (kt/mu)],
                  [1, 0, -1, 0, 0]])
   
   D = np.zeros((3,2))
   sys = StateSpace(A, B, C, D)
   if isdiscrete == True:
      if ts <= 0.0:
         raise ValueError(f'ts must be a positive float value, currently: %', ts)
      sys = sys.to_discrete(ts)
   return sys

def get_ss_dims(ss:StateSpace):
   x_dim = (ss.A).shape[0]
   u_dim = (ss.B).shape[1]
   y_dim = (ss.C).shape[0]
   return x_dim, u_dim, y_dim

def c2d_zoh_ss(ss_cont:StateSpace, time_step):
   '''
   This function calculates the zoh transformation from a continuous state space to discrete
   state space over the provided time_step. This method avoids the requirement of an invertible
   A matrix.
      - Continuous state space: x_dot = A*x + B*u
      - Discrete state space: x[k+1] = A*x[k] + B*u[k]
   The tranformation integrates the continuous state space over the time step, and sums the initial
   value with the integrated value to calculate the state value at the next time step. The control input
   is held constant over the interval, hence the zero-order hold.
   Algorithm:
      - Create combined A B matrix AB = [[A, B],[0,0]]
      - propagate state using e^(AB*dt) solution to the differential equation
      - break apart resulting matrix into corresponding discrete A B parts
      - pass C and D matrices through with no change
   '''
   n  = ss_cont.A.shape[0]
   m  = ss_cont.B.shape[1]
   A_B_c  = np.concatenate((np.concatenate((ss_cont.A, ss_cont.B),axis=1),np.zeros((m,n + m))), axis=0)
   A_B_d  = expm(A_B_c * time_step) 
   return StateSpace(A_B_d[:n,:n]  ,A_B_d[:n,n:] , ss_cont.C, ss_cont.D, dt = time_step)