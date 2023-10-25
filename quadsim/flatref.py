import numpy as np

from python_utils.polyu import deriv_fitting_matrix

deriv_names = ['pos', 'vel', 'acc', 'jerk', 'snap']
yawderiv_names = ['yaw', 'yawvel', 'yawacc']

class StaticRef:
  def __init__(self, pos, yaw=0.0):
    self.posdes = pos
    self.yawdes = yaw

    for name in deriv_names[1:]:
      setattr(self, name, lambda t: np.zeros((3,) + np.zeros_like(t).shape))
    for name in yawderiv_names[1:]:
      setattr(self, name, lambda t: np.zeros_like(t))

  def pos(self, t):
    # What's a better way here?
    if type(t) is np.ndarray:
      return np.tile(self.posdes, (len(t), 1)).T
    return self.posdes

  def yaw(self, t):
    return self.yawdes * np.ones_like(t)

class PosLine(StaticRef):
  """ Starts at rest and ends at rest. """
  def __init__(self, start, end, duration, yaw=0, initbase=True):
    if initbase:
      super().__init__(end, yaw=yaw)

    matfit = np.linalg.inv(deriv_fitting_matrix(8, t_end=duration))

    self.duration = duration

    polys = []
    for i in range(3):
      polys.append(matfit.dot(np.array((start[i], 0, 0, 0, end[i], 0, 0, 0)))[::-1])

    self.derivs = [polys]
    for i in range(4):
      add = []
      for poly in self.derivs[-1]:
        add.append(np.polyder(poly))
      self.derivs.append(add)

    for di, (d, name) in enumerate(zip(self.derivs, deriv_names)):
      def f(t, d=d, di=di, name=name):
        ret = []
        for axi in range(3):
          vals = np.polyval(d[axi], np.clip(t, 0.0, self.duration))
          # Zero out HODs past the end of the trajectory
          fill = vals if not di else 0.0
          ret.append(np.where(np.logical_or(t < 0, t > self.duration), fill, vals))

        return np.array(ret)

      setattr(self, name, f)

class YawLine(StaticRef):
  def __init__(self, pos, startyaw, endyaw, duration, initbase=True):
    if initbase:
      super().__init__(pos, endyaw)

    self.duration = duration

    matfit = np.linalg.inv(deriv_fitting_matrix(4, t_end=duration))
    yawpoly = matfit.dot(np.array((startyaw, 0, endyaw, 0)))[::-1]
    self.yawderivs = [yawpoly]

    for i in range(2):
      self.yawderivs.append(np.polyder(self.yawderivs[-1]))

    for di, (d, name) in enumerate(zip(self.yawderivs, yawderiv_names)):
      def f(t, d=d, di=di, name=name):
        vals = np.polyval(d, np.clip(t, 0.0, self.duration))
        # Zero out HODs past the end of the trajectory
        fill = vals if not di else 0.0
        return np.where(np.logical_or(t < 0, t > self.duration), fill, vals)

      setattr(self, name, f)

class PosLineYawLine(PosLine, YawLine):
  def __init__(self, start, end, duration, startyaw, endyaw):
    PosLine.__init__(self, start, end, duration, initbase=False)
    YawLine.__init__(self, end, startyaw, endyaw, duration, initbase=False)
