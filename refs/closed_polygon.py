import random
import numpy as np
import matplotlib.pyplot as plt
from DATT.refs.base_ref import BaseRef

class ClosedPoly(BaseRef):
    def __init__(self, speed=1, radius=1, sides =3, ang=None, direction=None, equal_division=False, random=False, seed=2023,env_diff_seed=False, fixed_seed=False):
        self.speed = speed
        self.radius = 0.5
        if random:
            self.sides = np.random.randint(3, 7)
            self.speed = np.random.uniform(0.9, 1.6)
            # print(self.sides, self.speed)
        else:
            self.sides = sides
        self.fixed_seed = fixed_seed
        self.seed = seed
        self.reset_count = 0
        self.env_diff_seed = env_diff_seed
        np.random.seed(self.seed)

        self.ang = ang
        if self.ang == None:
            ang = np.random.uniform(0, 2 * np.pi)
        # ang = 75
        self.center = self.radius * np.array([np.cos(ang), np.sin(ang)])
        self.start_point = np.zeros(2)

        self.equal_division = equal_division
        if self.equal_division:
            self.angle_with_center = np.linspace(0, 2 * np.pi, self.sides + 1)
        else:
            
            start, end = 0, 2 * np.pi
            divisions = self.sides
            points = sorted(np.random.uniform(start, end) for _ in range(divisions - 1))
    
            divisions = [start] + points + [end]

            self.angle_with_center = np.array(divisions)
        
        self.direction = direction
        if self.direction == None:
            # anticlockwise and clockwise
            direction = np.random.choice([1, -1])

        self.init_angle = np.arctan2((self.start_point[1] - self.center[1]) , (self.start_point[0] - self.center[0]))

        self.angle_with_center = direction * self.angle_with_center
        self.angle_with_center += self.init_angle

        self.get_vertices()

        self.reset()

    def get_vertices(self,):

        vertices = [self.start_point]
        for i in range(1, len(self.angle_with_center)):
            angle = self.angle_with_center[i]
            vert = self.center + self.radius * np.array([np.cos(angle),  
                                                        np.sin(angle)])
            
            vertices.append(vert)
        
        vertices = np.array(vertices)

        diff = np.diff(vertices, axis=0)
        self.side_lens = np.sqrt(np.sum(diff ** 2, axis=-1))
        self.perimeter = np.sum(self.side_lens)
        self.T = self.perimeter / self.speed

        self.t_per_side = self.side_lens / self.perimeter * self.T
        self.t_T = np.cumsum(self.t_per_side)

        self.v_per_side = diff / self.t_per_side[:, None]  
        self.verts = vertices      

               
    def reset(self, ):
        if self.fixed_seed: #self.reset_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            self.seed = random.randint(0, 1000000)
            np.random.seed(self.seed)

        # self.ang = ang
        if self.ang == None:
            ang = np.random.uniform(0, 2 * np.pi)
        # ang = 75
        self.center = self.radius * np.array([np.cos(ang), np.sin(ang)])
        self.start_point = np.zeros(2)

        # self.equal_division = equal_division
        if self.equal_division:
            self.angle_with_center = np.linspace(0, 2 * np.pi, self.sides + 1)
        else:
            
            start, end = 0, 2 * np.pi
            divisions = self.sides
            points = sorted(np.random.uniform(start, end) for _ in range(divisions - 1))
    
            divisions = [start] + points + [end]

            self.angle_with_center = np.array(divisions)
        
        # self.direction = direction
        if self.direction == None:
            # anticlockwise and clockwise
            direction = np.random.choice([1, -1])

        self.init_angle = np.arctan2((self.start_point[1] - self.center[1]) , (self.start_point[0] - self.center[0]))

        self.angle_with_center = direction * self.angle_with_center
        self.angle_with_center += self.init_angle

        self.get_vertices()
        self.reset_count += 1

    def pos(self, t):
        x = 0 * t
        y = 0 * t
        z = 0 * t

        for i in range(self.sides):
            if i!=self.sides-1:
                t_mask = ((t % self.T) < self.t_T[i])
                if i != 0:
                    t_mask *= (self.t_T[i - 1] <= (t % self.T))
            else:
                t_mask = (self.t_T[i - 1] <= (t % self.T))

            t_ = (t % self.T)
            if i!=0:
                t_ -= self.t_T[i - 1]

            x += t_mask * (self.v_per_side[i][0] * t_ + self.verts[i][0])
            y += t_mask * (self.v_per_side[i][1] * t_ + self.verts[i][1])
        
        return np.array([x, y, z])

    def vel(self, t):
        x = 0 * t
        y = 0 * t
        z = 0 * t

        for i in range(self.sides):
            if i!=self.sides-1:
                t_mask = ((t % self.T) < self.t_T[i])
                if i != 0:
                    t_mask *= (self.t_T[i - 1] <= (t % self.T))
            else:
                t_mask = (self.t_T[i - 1] <= (t % self.T))

            t_ = (t % self.T)
            if i!=0:
                t_ -= self.t_T[i - 1]

            x += t_mask * self.v_per_side[i][0]
            y += t_mask * self.v_per_side[i][1]
        
        return np.array([x, y, z])
    
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

if __name__== "__main__":
    cp = ClosedPoly(random=True, seed=np.random.randint(0, 10000000))
    # cp.reset()
    t = np.linspace(0, 10.0, 100)
    pos = cp.pos(t)
    vel = cp.vel(t)
    plt.plot(pos[0], pos[1])
    plt.scatter(cp.verts[:, 0], cp.verts[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t, pos[0])
    plt.plot(t, vel[0])

    plt.subplot(3, 1, 2)
    plt.plot(t, pos[1])
    plt.plot(t, vel[1])

    plt.subplot(3, 1, 3)
    plt.plot(t, pos[2])
    plt.plot(t, vel[2])

    plt.show()

# import pdb;pdb.set_trace()
# k  = a.get_vertices()
# plt.plot(k[:, 0], k[:, 1])
# plt.show()
# import pdb;pdb.set_trace()
# start = 0.0
# end = 100.0
# divisions = 5

# divisions, intervals = divide_range(start, end, divisions)
# print("Divisions:", divisions)
# print("Intervals:", intervals)