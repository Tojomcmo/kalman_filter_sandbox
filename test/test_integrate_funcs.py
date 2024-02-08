import unittest
import numpy as np
import src.integrate_funcs as int_funcs

class dynamics_rk4_zoh_tests(unittest.TestCase):
    def dyn_func(self, x, u):
        x_kp1 = np.zeros((2,1),dtype=float)
        x_kp1[0] = 1*x[0] + 1*x[1] + 1*u[0] + 1*u[1]
        x_kp1[1] = 2*x[0] + 2*x[1] + 2*u[0] + 2*u[1]
        return x_kp1

    def test_dynamics_rk4_zoh_generates_valid_output(self):
        ts = 0.1
        x = np.array([[1],[2]])
        u = np.array([[1],[2]])
        x_kp1_test = int_funcs.dynamics_rk4_zoh(self.dyn_func, ts, x, u)
        self.assertEqual(x_kp1_test.shape, x.shape)

if __name__ == '__main__':
    unittest.main()