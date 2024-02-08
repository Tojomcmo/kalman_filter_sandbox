import numpy as np
from scipy.signal import StateSpace as ss



def qc_ss_1(params):
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
                  [1, 0]])
    
    C = np.array([[-(ks/ms), -(bs/ms), (ks/ms), (bs/ms), 0],
                  [(ks/mu), (bs/mu), -((ks+kt)/mu), -(bs/mu), (kt/mu)],
                  [1, 0, -1, 0, 0]])
    
    D = np.zeros((3,2))

    return ss(A,B,C,D)



def qc_dyn_1(x, u):
    # This function generates the state space model for
    # x-[in]: state vector [5x1] 
    #    zs     - sprung mass position
    #    zs_dot - sprung mass velocity
    #    zu     - unsprung mass position
    #    zu_dot - unsprung mass velocity
    #    zr     - road profile position (vertical)
    # u-[in]: input vector [2x1]
    #    fs     - active suspension force
    #    zr_dot - road profile velocity (vertical)


    return