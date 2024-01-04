from DATT.refs.chained_poly_ref import ChainedPolyRef
from DATT.refs.polynomial_ref import PolyRef
from DATT.refs.random_zigzag import RandomZigzag
from DATT.refs.setpoint_ref import SetpointRef
from DATT.refs.pointed_star import NPointedStar
from DATT.refs.closed_polygon import ClosedPoly
from DATT.refs.base_ref import BaseRef

import numpy as np
import random

class MixedTrajectoryRef(BaseRef):
    def __init__(self, altitude, ymax=0.0, zmax=0.0, diff_axis=False, include_all=False, seed=2023, env_diff_seed=False, init_ref=None, fixed_seed=False, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.poly_ref = PolyRef(altitude=altitude, use_y=ymax > 0.0, seed=seed, fixed_seed=fixed_seed)
        # print('DIFF AXIS', diff_axis)
        self.zigzag_ref = RandomZigzag(max_D=[1.0, ymax, zmax], diff_axis=diff_axis, seed=seed, fixed_seed=fixed_seed)
        self.chained_poly_ref = ChainedPolyRef(altitude=altitude, use_y=ymax > 0.0, seed=seed, fixed_seed=fixed_seed)
        self.setpoint_ref = SetpointRef(randomize=True)
        self.star = NPointedStar(radius=1, random=True, env_diff_seed=env_diff_seed, seed=seed, fixed_seed=fixed_seed)
        self.closed_poly = ClosedPoly(random=True)
        self.init_ref = init_ref
        self.reset_count = 0
        self.env_diff_seed = env_diff_seed

        if include_all:
            self.refs = [self.poly_ref, self.zigzag_ref, self.chained_poly_ref, self.star, self.closed_poly]
        else:
            self.refs = [self.poly_ref, self.zigzag_ref, self.chained_poly_ref]

        # print('seed', seed)
        self.seed = seed
        self.fixed_seed = fixed_seed
        np.random.seed(self.seed)
        self.reset()

        if self.init_ref == -1:
            # print('INIT SETPOINT REF')
            self.curr_ref = self.setpoint_ref
            self.curr_ref.setpoint = np.array([0, 0, 0])
        if self.init_ref is not None:
            print('init_ref', self.init_ref, self.refs[self.init_ref].__class__)
            self.curr_ref = self.refs[self.init_ref]

    def reset(self):
        if self.fixed_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            np.random.seed(random.randint(0, 1000000))

        self.curr_ref = np.random.choice(self.refs)

        # print('reset', self.curr_ref.__class__)

        self.poly_ref.reset()
        self.zigzag_ref.reset()
        self.chained_poly_ref.reset()
        self.setpoint_ref.reset()
        self.star.reset()

        self.reset_count += 1

    def pos(self, t):
        return self.curr_ref.pos(t)
    
    def vel(self, t):
        return self.curr_ref.vel(t)
    
    def acc(self, t):
        return self.curr_ref.acc(t)

    def jerk(self, t):
        return self.curr_ref.jerk(t)

    def snap(self, t):
        return self.curr_ref.snap(t)

    def yaw(self, t):
        return self.curr_ref.yaw(t)
    
    def yawvel(self, t):
        return self.curr_ref.yawvel(t)

    def yawacc(self, t):
        return self.curr_ref.yawacc(t)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ref = MixedTrajectoryRef(altitude=1.0, ymax=1.0, seed=np.random.randint(10000))
    t = np.linspace(0, 10, 500)

    plt.subplot(2, 1, 1)
    plt.plot(t, ref.pos(t)[0, :], label='x')
    plt.subplot(2, 1, 2)
    plt.plot(t, ref.pos(t)[1, :], label='y')
    plt.show()
