import unittest
import numpy as np
from scipy.signal import StateSpace

import src.dyn_funcs as dyn_funcs

class qc_ss_1_tests(unittest.TestCase):
    def test_qc_ss_1_generates_valid_continuous_state_space(self):
        params = {
                    'ms'  : 1,   
                    'mu'  : 2,
                    'bs'  : 3,
                    'ks'  : 4,
                    'kt'  : 5,
        }
        qc_ss_test = dyn_funcs.qc_ss_1(params)

        A_valid = np.array([[0, 1, 0, 0, 0],
                      [-4, -3, 4, 3, 0],
                      [0, 0, 0, 1, 0],
                      [4/2, 3/2, -9/2, -3/2, 5/2],
                      [0, 0, 0, 0, 0]])
        B_valid = np.array([[0, 0],
                      [1, 0],
                      [0, 0],
                      [-1/2, 0],
                      [0, 1]])
        C_valid = np.array([[-4, -3, 4, 3, 0],
                      [4/2, 3/2, -9/2, -3/2, 5/2],
                      [1, 0, -1, 0, 0]])
        D_valid = np.array([[0, 0],
                            [0, 0],
                            [0, 0]])
        self.assertEqual((qc_ss_test.A).tolist(), A_valid.tolist())
        self.assertEqual((qc_ss_test.B).tolist(), B_valid.tolist())
        self.assertEqual((qc_ss_test.C).tolist(), C_valid.tolist())
        self.assertEqual((qc_ss_test.D).tolist(), D_valid.tolist())

    def test_qc_ss_1_generates_valid_discrete_zoh_state_space(self):
        params = {
                    'ms'  : 1,   
                    'mu'  : 2,
                    'bs'  : 3,
                    'ks'  : 4,
                    'kt'  : 5,
        }
        ts = 0.1
        qc_ss_d_test = dyn_funcs.qc_ss_1(params, isdiscrete=True, ts=ts)

        A_valid = np.array([[0, 1, 0, 0, 0],
                      [-4, -3, 4, 3, 0],
                      [0, 0, 0, 1, 0],
                      [4/2, 3/2, -9/2, -3/2, 5/2],
                      [0, 0, 0, 0, 0]])
        B_valid = np.array([[0, 0],
                      [1, 0],
                      [0, 0],
                      [-1/2, 0],
                      [0, 1]])
        C_valid = np.array([[-4, -3, 4, 3, 0],
                      [4/2, 3/2, -9/2, -3/2, 5/2],
                      [1, 0, -1, 0, 0]])
        D_valid = np.array([[0, 0],
                            [0, 0],
                            [0, 0]])
        qc_ss_d_valid = dyn_funcs.c2d_zoh_ss(StateSpace(A_valid, B_valid, C_valid, D_valid), ts)

        self.assertEqual((qc_ss_d_test.A).tolist(), qc_ss_d_valid.A.tolist())
        self.assertEqual((qc_ss_d_test.B).tolist(), qc_ss_d_valid.B.tolist())
        self.assertEqual((qc_ss_d_test.C).tolist(), qc_ss_d_valid.C.tolist())
        self.assertEqual((qc_ss_d_test.D).tolist(), qc_ss_d_valid.D.tolist())

if __name__ == '__main__':
    unittest.main()