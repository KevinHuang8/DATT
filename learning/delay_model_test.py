import matplotlib.pyplot as plt
import numpy as np


# data = np.load('delay_model.npz', mmap_mode='r')

# for f in data.files:
#     print(f)


u = 10
kx = 0.4
kz = 0.4

t_end = 1
dt = 0.02

def run(kx, kz, first=False):
    all_x = []
    x = 0
    z = 0
    for t in np.arange(0, t_end, dt):
        # first order time delay system
        if first:
            dx = kx * (u - x)
            x += dx
            all_x.append(x)
        else:
            # second order
            z_des = kx * (u - x)
            dz = kz * (z_des - z)
            z += dz
            x += z
            all_x.append(x)
    return all_x

# all_x = run(kx=0.4, kz=0.4)
# all_x2 = run(kx=0.4, kz=0.4, first=True)
# all_x3 = run(kx=0.6, kz=0.4)
# all_x4 = run(kx=0.4, kz=0.6)
# all_x5 = run(kx=0.6, kz=0.6)

all_x = run(kx=0.4, kz=1.0)
all_x2 = run(kx=0.4, kz=0.4, first=True)
all_x3 = run(kx=0.3, kz=0.6)
all_x4 = run(kx=0.4, kz=0.6)
all_x5 = run(kx=0.4, kz=0.4)

n = np.arange(0, t_end, dt).shape[0]
plt.plot(np.arange(0, t_end, dt), all_x, label='kx 0.4, kz 0.4')
plt.plot(np.arange(0, t_end, dt), np.ones(n) * u)
plt.plot(np.arange(0, t_end, dt), all_x2, label='first')
plt.plot(np.arange(0, t_end, dt), all_x3, label='kx 0.6, kz 0.4')
plt.plot(np.arange(0, t_end, dt), all_x4, label='kx 0.4, kz 0.6')
plt.plot(np.arange(0, t_end, dt), all_x5, label='kx 0.6, kz 0.6')


plt.legend()    
plt.show()