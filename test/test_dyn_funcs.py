import unittest
import numpy as np

import src.dyn_funcs as dyn_funcs

class qc_ss_1_tests(unittest.TestCase):
    def test_qc_ss_1_generates_valid_state_space(self):
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
                      [1, 0]])
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

if __name__ == '__main__':
    unittest.main()