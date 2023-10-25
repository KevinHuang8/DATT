import numpy as np

from scipy.spatial.transform import Rotation as R

from DATT.python_utils.mathu import e1, e2, e3, normalized

from DATT.quadsim.models import general_params

g = -e3 * general_params()['g']

def a_from_z(z, u):
  return u * z + g

def j_from_zdot(z, u, udot, zdot):
  return udot * z + u * zdot

def s_from_zddot(z, u, udot, uddot, zdot, zddot):
  return uddot * z + 2 * udot * zdot + u * zddot

def yaw_zyx_from_x(x):
  return np.arctan2(x[1], x[0])

def yawdot_zyx_from_xdot(x, xdot):
  x_xy_norm = x[0] ** 2 + x[1] ** 2
  if x_xy_norm < 1e-8:
    return 0.0

  return (-x[1] * xdot[0] + x[0] * xdot[1]) / x_xy_norm

def rot_from_z_yaw_zyx(z, yaw):
  # ZYX Euler angles yaw
  z = normalized(z)

  y_c = np.array((-np.sin(yaw), np.cos(yaw), 0))
  x_b = normalized(np.cross(y_c, z))
  y_b = np.cross(z, x_b)

  return R.from_matrix(np.column_stack((x_b, y_b, z)))

def rot_from_z_yaw_zxy(z, yaw):
  # ZXY Euler angles yaw
  z = normalized(z)

  x_c = np.array((np.cos(yaw), np.sin(yaw), 0))
  y_b = normalized(np.cross(z, x_c))
  x_b = np.cross(y_b, z)

  return R.from_matrix(np.column_stack((x_b, y_b, z)))

def rot_from_a_yaw_zyx(a, yaw):
  return rot_from_z_yaw_zyx(z_from_a(a), yaw)

def u1_from_a(a):
  return np.linalg.norm(a - g)

def z_from_a(a):
  return (a - g) / u1_from_a(a)

def udot_from_j(j, z):
  return j.dot(z)

def zdot_from_j(u, j, z):
  return (1.0 / u) * (j - udot_from_j(j, z) * z)

def uddot_from_s(u, s, z, zdot):
  return s.dot(z) + u * zdot.dot(zdot)

def uddot_from_flat(u, a, j, s):
  z = z_from_a(a)
  zdot = zdot_from_j(u, j, z)
  return uddot_from_s(u, s, z, zdot)

def zddot_from_s(u, j, s, z):
  zdot = zdot_from_j(u, j, z)
  udot = udot_from_j(j, z)
  return (1.0 / u) * (s - 2 * udot * zdot - uddot_from_s(u, s, z, zdot) * z)

def omega_from_zdot(R, zdot, xdot):
  # See thesis proposal.
  y = R.apply(e2)
  z = R.apply(e3)
  return np.cross(z, zdot) + y.dot(xdot) * z

def omega_from_flat(u, a, j, yaw, yawdot):
  z = z_from_a(a)
  rot = rot_from_z_yaw_zyx(z, yaw)
  zdot = zdot_from_j(u, j, z)
  xdot, _ = get_xdot_xddot(yawdot, 0, rot, zdot, 0 * zdot)
  return omega_from_zdot(rot, zdot, xdot)

def omega_from_flat_2(u, a, j, yaw, yawdot):
  z = z_from_a(a)
  y_C = np.array((-np.sin(yaw), np.cos(yaw), 0))
  omega2 = (1.0 / u) * np.cross(z, j) + (1.0 / np.linalg.norm(np.cross(y_C, z)) ** 2) * ( yawdot * z.dot(e3) + (1.0 / u) * y_C.dot(z) * y_C.dot(np.cross(z, j))) * z

def alpha_from_zddot(R, zdot, zddot, omega, xddot):
  # See thesis proposal.
  x = R.apply(e1)
  y = R.apply(e2)
  z = R.apply(e3)
  return np.cross(z, zddot) + omega.dot(z) * zdot + (y.dot(xddot) - omega.dot(x) * omega.dot(y)) * z

def alpha_from_flat(u, a, j, s, yaw, yawdot, yawddot):
  return att_hod_from_flat(u, a, j, s, yaw, yawdot, yawddot)[2]

def att_hod_from_flat(u, a, j, s, yaw, yawdot, yawddot):
  z = z_from_a(a)
  rot = rot_from_z_yaw_zyx(z, yaw)
  zdot = zdot_from_j(u, j, z)
  zddot = zddot_from_s(u, j, s, z)
  xdot, xddot = get_xdot_xddot(yawdot, yawddot, rot, zdot, zddot)
  omega = omega_from_zdot(rot, zdot, xdot)
  return rot, omega, alpha_from_zddot(rot, zdot, zddot, omega, xddot)

def angvel_hod_from_flat(u, a, j, yaw, yawdot):
  """ Return rot, omega, Euler ZYX for yaw """
  z = z_from_a(a)
  rot = rot_from_z_yaw_zyx(z, yaw)
  zdot = zdot_from_j(u, j, z)
  xdot = get_xdot(yawdot, rot, zdot)
  omega = omega_from_zdot(rot, zdot, xdot)
  return rot, omega

def _get_xdot_Ainv(yawvel, R, zdot):
  """ Euler ZYX for yaw """
  # Solve for angvel z using three constraints
  # (1) x2 / x1 = tan(yaw)
  # (2) x dot z = 0
  # (3) x dot x = 1

  x = R.apply(e1)
  z = R.apply(e3)

  x_xy_norm2 = x[0] ** 2 + x[1] ** 2

  A = np.zeros((3, 3))
  A[0, :] = np.array((-x[1], x[0], 0))
  A[1, :] = z
  A[2, :] = x

  Ainv = np.linalg.inv(A)

  b = np.array((yawvel * x_xy_norm2, -x.dot(zdot), 0))
  xdot = Ainv.dot(b)

  return xdot, Ainv

def get_xdot(yawvel, R, zdot):
  return _get_xdot_Ainv(yawvel, R, zdot)[0]

def get_xdot_xddot(yawvel, yawacc, R, zdot, zddot):
  """ Return (xdot, xddot) in world frame (same frame as args?)
      for Euler ZYX
  """

  xdot, Ainv = _get_xdot_Ainv(yawvel, R, zdot)

  x = R.apply(e1)
  z = R.apply(e3)

  x_xy_norm2 = x[0] ** 2 + x[1] ** 2

  if False:
    yaw = np.arctan2(x[1], x[0])
    x_C = np.array(( np.cos(yaw), np.sin(yaw), 0))
    y_C = np.array((-np.sin(yaw), np.cos(yaw), 0))
    xdot_ana = (np.eye(3) - np.outer(x, x)).dot(-yawvel * np.cross(x_C, z) + np.cross(y_C, zdot)) / np.linalg.norm(np.cross(y_C, z))

    assert np.allclose(xdot_ana, xdot)

  # Solve for xddot using the same three constraints
  # It turns out that the A matrix is the same.
  b20 = yawacc * x_xy_norm2 + 2 * yawvel * (x[0] * xdot[0] + x[1] * xdot[1])
  b2 = np.array((b20, -x.dot(zddot) -2 * xdot.dot(zdot), -xdot.dot(xdot)))
  xddot = Ainv.dot(b2)

  return xdot, xddot
