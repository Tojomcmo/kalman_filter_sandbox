import numpy as np

class SingleRoadFeature:
    def __init__(self, x_start, height, width, profile='halfsine'):
        self.x_start = x_start
        self.height = height
        self.width = width
        self.profile = profile
        self.x_end = x_start + width

    def get_z_pos(self, x_pos):
        if x_pos <= self.x_start or x_pos >= self.x_end:
            z_pos = 0   
        else:
            if self.profile == 'halfsine':
                z_pos = self.height * np.sin(np.pi * (x_pos - self.x_start) / self.width)    
            elif self.profile == 'rectangle':    
                z_pos = self.height 
        return z_pos

    def get_z_vel(self, x_pos, x_vel):
        # grab time_step that normalizes the slope calculation step to a subdivision of the bump width
        x_step = self.width * 10e-3
        t_step = x_step/x_vel
        x_pp1 = self.get_z_pos(x_pos + x_step)
        x_pm1 = self.get_z_pos(x_pos - x_step)
        return (x_pp1 - x_pm1)/(2*t_step)

    def create_profile_array(self, x_vec):
        return np.array([self.get_z_pos(xi) for xi in x_vec])  
    

class CombinedRoadProfile:
    def __init__(self):
        self.feature_list = []

    def add_new_feature(self, new_feature:SingleRoadFeature):  
        self.feature_list.append(new_feature)

    def get_z_pos(self, x_pos):
        if len(self.feature_list)==0:
            raise ValueError('no road features added to profile, please add a road feature')
        z_pos = 0.0
        for feature in self.feature_list:
            z_pos += feature.get_z_pos(x_pos)
        return z_pos
    
    def get_z_vel(self, x_pos, x_vel):
        # grab time_step that normalizes the slope calculation step to a subdivision of the bump width
        x_step = self.width * 10e-3
        t_step = x_step/x_vel
        x_pp1 = self.get_z_pos(x_pos + x_step)
        x_pm1 = self.get_z_pos(x_pos - x_step)
        return (x_pp1 - x_pm1)/(2*t_step)      
      
    def create_profile_array(self, x_vec):
        return np.array([self.get_z_pos(xi) for xi in x_vec])  