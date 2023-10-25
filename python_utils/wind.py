import numpy as np

from DATT.python_utils.mathu import normalized
from DATT.python_utils.plotu import set_3daxes_equal

class WindModel:
  def __init__(self, pos, direction, vmax, radius, decay_lat=4, decay_long=0.3, dispangle=np.radians(15)):
    self.pos = pos
    self.dir = normalized(direction)
    self.vmax = vmax
    self.radius = radius
    self.decay_lat = decay_lat
    self.decay_long = decay_long
    self.coneslope = np.tan(dispangle)

  def velocity(self, pos):
    """
      The fan velocity is radially symmetric around the fan,
      has a maximum of vmax and decays exponentially in two directions:
        perpendicular to the fan direction and
        parallel to the fan direction

      decay_lat controls how fast it decays perpendicularly
        if set to 0, results in an infinitely large wall of wind
      decay_long for parallel
        if set to 0, results in an infinitely long cone of wind

      radius denotes a disk perpendicular to the direction over which there is no decay

      Ideally decay_lat is large and > decay_long, to get a sharp cutoff at the fan edge (edge of disk)

      dispangle controls at what angle the radius grows, i.e. simulates wind field dispersion
      if set to 0, wind beam will not expand

      Ideally, this can be made more physically correct by solving some boundary value diff eq.
    """

    dist = pos - self.pos
    r_para = dist.dot(self.dir)

    if r_para < 0:
      return 0

    r_perp = np.linalg.norm(np.cross(dist, self.dir))

    r_eff = self.radius + self.coneslope * r_para
    r_perp = max(0, r_perp - r_eff)

    return self.vmax * np.exp(-self.decay_lat * r_perp) * np.exp(-self.decay_long * r_para) * self.dir

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  pos = np.array((0, 0., 0))
  direction = np.array((1, 0.0, 0.0))
  model = WindModel(pos, direction, vmax=10.0, radius=0.2, decay_lat=4, decay_long=0.3)

  nx = 6
  ny = 12
  nz = 12

  plot_rad = 3
  plot_len = 10

  x, y, z = np.meshgrid(
    np.linspace(0, plot_len, nx),
    np.linspace(-plot_rad, plot_rad, ny),
    np.linspace(-plot_rad, plot_rad, nz)
  )

  output = np.zeros((ny, nx, nz, 3))
  for i in range(ny):
    for j in range(nx):
      for k in range(nz):
        output[i, j, k] = model.velocity(np.array((x[i, j, k], y[i, j, k], z[i, j, k])))

  fig = plt.figure("Wind Vector Field")
  ax = fig.gca(projection='3d')
  ax.quiver(x, y, z, output[:, :, :, 0], output[:, :, :, 1], output[:, :, :, 2], length=0.4)
  ax.set_title("Wind Vector Field")
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  set_3daxes_equal(ax)
  plt.show()
