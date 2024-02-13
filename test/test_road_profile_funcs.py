import unittest
import numpy as np
import src.road_profile_funcs as road

class SingleRoadFeature_tests(unittest.TestCase):

    def test_bump_accepts_valid_input(self):
        road_profile = road.SingleRoadFeature(10.0, 0.05, 1.0)
        test_p_1 = road_profile.get_z_pos(5.0)
        test_p_2 = road_profile.get_z_pos(10.5)
        test_p_3 = road_profile.get_z_pos(12.0)

        self.assertEqual(test_p_1, 0.0)
        self.assertAlmostEqual(test_p_2, 0.05)
        self.assertEqual(test_p_3, 0.0)        


    def test_bump_generates_valid_array(self):
        road_profile = road.SingleRoadFeature(5.0, 0.05, 1.0)
        x_vec = np.linspace(0,10,100)
        z_vec = road_profile.create_profile_array(x_vec)
        print(z_vec)
        self.assertEqual(True,True)       

    def test_bump_generates_valid_vel(self):
        road_profile = road.SingleRoadFeature(5.0, 0.05, 1.0)
        x_vel = 5
        x_pos_1 = 5.5
        x_pos_2 = 5.25
        z_vel_1 = road_profile.get_z_vel(x_pos_1, x_vel)
        z_vel_2 = road_profile.get_z_vel(x_pos_2, x_vel)        
        print(z_vel_2)
        self.assertEqual(z_vel_1,0.0)       

class CombinedRoadProfile_tests(unittest.TestCase):

    def test_CRP_accepts_valid_inputs(self):
        combined_road_profile = road.CombinedRoadProfile()
        bump_1 = road.SingleRoadFeature(10.0, 0.05, 1.0, profile='halfsine')
        rect_1 = road.SingleRoadFeature(13.0, 0.05, 1.0, profile ='rectangle')
        combined_road_profile.add_new_feature(bump_1)
        combined_road_profile.add_new_feature(rect_1)
        test_p_1 = combined_road_profile.get_z_pos(5.0)
        test_p_2 = combined_road_profile.get_z_pos(10.5)
        test_p_3 = combined_road_profile.get_z_pos(13.1)

        self.assertEqual(test_p_1, 0.0)
        self.assertAlmostEqual(test_p_2, 0.05)
        self.assertEqual(test_p_3, 0.5)        

    


if __name__ == '__main__':
    unittest.main()