import numpy as np
import matplotlib.pyplot as plt
from DATT.quadsim.rigid_body import State_struct
from scipy.spatial.transform import Rotation as R

class BaseRef():
    def ref_vec(self, t):
        pos = self.pos(t)
        vel = self.vel(t)
        quat = self.quat(t)
        omega = self.angvel(t)

        if isinstance(t, np.ndarray):
            refVec = np.vstack((pos, vel, quat, omega))
        else:
            refVec = np.r_[pos, vel, quat, omega]
        return refVec

    def get_state_struct(self, t):
        
        return State_struct(
            pos = self.pos(t),
            vel = self.vel(t),
            acc = self.acc(t),
            jerk = self.jerk(t),
            snap = self.snap(t),
            rot = R.from_quat(self.quat(t)),
            ang = self.angvel(t),
        )
        
    def pos(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    
    def vel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    
    def acc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def jerk(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def snap(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    def quat(self, t):
        '''
        w,x,y,z
        '''
        return np.array([
            t ** 0,
            t * 0,
            t * 0,
            t * 0
        ])
    def angvel(self, t):
        return np.array([
            t * 0,
            t * 0,
            t * 0,
        ])
    def yaw(self, t):
        return t * 0

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

if __name__=='__main__':
    a = BaseRef()
    t = np.array([0., 1.0])
    a.ref_vec(t)
