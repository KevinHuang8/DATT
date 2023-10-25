import numpy as np
import random
import matplotlib.pyplot as plt

class NPointedStar:
    def __init__(self, n_points=5, speed=3, radius=1, random=False, seed=2023, env_diff_seed=False, fixed_seed=False):
        self.n_points = n_points
        self.speed = speed
        self.radius = radius
        self.random=random  
        self.reset_count = 0
        self.env_diff_seed = env_diff_seed

        # print('seed', seed)
        self.seed = seed
        self.fixed_seed = fixed_seed
        np.random.seed(self.seed)
        self.reset()

    def reset(self):
        if self.fixed_seed: #self.reset_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            seed = random.randint(0, 1000000)
            np.random.seed(seed)

        if self.random:
            self.n_points = 2*np.random.randint(2, 6) + 1
            self.speed = np.random.uniform(0.8, 2.2)

        self.speed = 1.0
        self.n_points = 5
        self.radius = 0.7

        theta = 0
        d_theta = 2*np.pi/self.n_points

        x = 0#-self.radius
        y = 0

        self.points = []

        for i in range(self.n_points):
            new_x = x + self.radius*np.cos(theta)
            new_y = y + self.radius*np.sin(theta)
            theta += d_theta
            
            self.points.append((new_x, new_y))

        self.points = np.array(self.points)

        angle_diff = np.pi/self.n_points

        chord_angle = np.pi - angle_diff

        total_time = (2 * self.radius * np.sin(chord_angle/2) * self.n_points) / self.speed

        self.time_to_start = 1

        self.T = np.arange(0, 100, total_time / self.n_points)

        self.T = np.r_[0, self.T + self.time_to_start]

        self.reset_count += 1

    def pos(self, t):
        i = np.array(np.searchsorted(self.T, t))

        i[i == 0] = 1

        init = i == 1

        pointA = np.copy(self.points[((i - 2)*(self.n_points // 2)) % self.n_points])
        pointB = self.points[((i - 1)*(self.n_points // 2)) % self.n_points]

        pointA[init] = np.zeros((np.where(init)[0].shape[0], 2))

        tA = self.T[i - 1]
        tB = self.T[i]

        # parametric line connecting points
        if isinstance(t, np.ndarray):
            x = pointA[:, 0] + (pointB[:, 0] - pointA[:, 0]) / (tB - tA) * (t - tA)
            y = pointA[:, 1] + (pointB[:, 1] - pointA[:, 1]) / (tB - tA) * (t - tA)
        else:
            x = pointA[0] + (pointB[0] - pointA[0]) / (tB - tA) * (t - tA)
            y = pointA[1] + (pointB[1] - pointA[1]) / (tB - tA) * (t - tA)

        return np.array([x, y, t*0])
    
    def vel(self, t):
        i = np.array(np.searchsorted(self.T, t))

        i[i == 0] = 1

        init = i == 1

        pointA = np.copy(self.points[((i - 2)*(self.n_points // 2)) % self.n_points])
        pointB = self.points[((i - 1)*(self.n_points // 2)) % self.n_points]

        pointA[init] = np.zeros((np.where(init)[0].shape[0], 2))

        tA = self.T[i - 1]
        tB = self.T[i]

        # parametric line connecting points
        if isinstance(t, np.ndarray):
            x = (pointB[:, 0] - pointA[:, 0]) / (tB - tA) 
            y = (pointB[:, 1] - pointA[:, 1]) / (tB - tA)
        else:
            x = (pointB[0] - pointA[0]) / (tB - tA)
            y = (pointB[1] - pointA[1]) / (tB - tA)

        
        return np.array([x, y, t*0])

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

if __name__ == "__main__":
    ref = NPointedStar(n_points=5, speed=3, radius=1, random=True, seed=random.randint(0, 1000000))
    t = np.linspace(0, 20, 100)
    x = ref.pos(t)
    v = ref.vel(t)
    plt.subplot(3, 1, 1)
    plt.plot(t, x[0])
    plt.plot(t, v[0], label='vel')
    plt.subplot(3, 1, 2)
    plt.plot(t, x[1])
    plt.plot(t, v[1], label='vel')
    plt.subplot(3, 1, 3)
    plt.plot(t, x[2])
    plt.plot(t, v[2], label='vel')

    #plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[0], x[1], x[2])
    ax.scatter(ref.points[:, 0], ref.points[:, 1], np.zeros(ref.points.shape[0]), c='r', marker='o')

    plt.show()
