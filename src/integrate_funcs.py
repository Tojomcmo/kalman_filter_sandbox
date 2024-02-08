import numpy as np



def dynamics_rk4_zoh(dyn_func, ts, x, u):
    f1 = dyn_func(x, u)
    f2 = dyn_func(x + (ts/2)*f1, u)
    f3 = dyn_func(x + (ts/2)*f2, u)
    f4 = dyn_func(x + ts*f3, u)
    return x + ((ts/6) * (f1 + (f2/2) + (f3/2) + f4))