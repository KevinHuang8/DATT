import numpy as np
import matplotlib.pyplot as plt
from DATT.quadsim.rigid_body import State_struct
from scipy.spatial.transform import Rotation as R
from DATT.refs.base_ref import BaseRef

class takeofflanding_ref(BaseRef):

    def __init__(self, dest=None, rate=None, start_state=None, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.dest = dest
        self.rate = rate

        self.track_radius = 0.3
        self.start_state = start_state.pos
        
    def pos(self, t):
        # assert len(t.shape) == 1, 'time should be scalar for takeoff and landing'
        
        virt_curr_pos = self.start_state
        vector = self.dest - virt_curr_pos

        if np.linalg.norm(vector) > self.track_radius:
            vector = vector / np.linalg.norm(vector)

            vel = vector * self.rate

            track_pt = self.start_state + vel * 0.02
            self.start_state = track_pt.copy()

            return track_pt + self.offset_pos

        else:
            return self.dest + self.offset_pos

