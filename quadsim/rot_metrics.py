import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from python_utils.mathu import vee, e1, e3, skew_matrix, normang
from python_utils.plotu import named

from scipy.spatial.transform import Rotation as R

#r1 = R.from_euler('ZYX', [0.4, 0.5, 1.4])
#r2 = R.from_euler('ZYX', [0.3, 0.2, 0.0])

def euler(s, r1, r2):
  r1 = r1.as_euler(s)
  r2 = r2.as_euler(s)
  return normang((r1 - r2)[::-1])

euler_zyx = partial(euler, 'ZYX')

def roterr(r1, r2):
  return r2.T.dot(r1)
  #return r2.dot(r1.T)
  #return r1.dot(r2.T)
  #return r1.T.dot(r2)

def matf(r1, r2):
  rerr = roterr(r1, r2)
  return vee(rerr - rerr.T)

def so3_discon(r1, r2):
  """ As in 2013 T. Lee paper: Exponential stability of an att. ... """
  r1 = r1.as_matrix()
  r2 = r2.as_matrix()

  Re = roterr(r1, r2)

  # I removed a factor of 2 in the denominator here to match the quat error function
  return (1.0 / np.sqrt(1 + np.trace(Re))) * vee(Re - Re.T)

def so3_con(r1, r2):
  """ As in 2010 T. Lee paper: Geometric Tracking Control of q Quadrotor UAV ... """
  r1 = r1.as_matrix()
  r2 = r2.as_matrix()

  return (1.0 / 2) * matf(r1, r2)

def quat(r1, r2):
  """ As in 2013 Fresk and Nikolakopoulos """
  errq = (r2.inv() * r1).as_quat()
  return 2 * np.sign(errq[3]) * errq[0:3]

def rotvec(r1, r2):
  #errq = (r2.inv() * r1).as_quat()
  #thby2 = np.arccos(np.abs(errq[3]))
  # (thby2 / np.sin(thby2)) * 2 * np.sign(errq[3]) * errq[0:3]
  return (r2.inv() * r1).as_rotvec()

def rotvec_tilt_priority(r1, r2, k_yaw=0.5):
  """ As in 2018 Mueller's Multicopter attitude control for recovery from large disturbances
      k_yaw should be between 0 and 1 """
  # TODO Don't use both quat and rots here.
  Re = roterr(r1.as_matrix(), r2.as_matrix())
  errq = (r2.inv() * r1).as_quat()

  ang_reduced = np.arccos(e3.T.dot(Re.dot(e3)))
  ax_reduced = skew_matrix(Re.T.dot(e3)).dot(e3) / np.sin(ang_reduced)

  thby2 = np.arccos(np.abs(errq[3]))
  rv = (thby2 / np.sin(thby2)) * 2 * np.sign(errq[3]) * errq[0:3]

  k_reduced = 1.0
  return k_yaw * rv + (k_reduced - k_yaw) * ang_reduced * ax_reduced

def rotvec_tilt_priority2(r1, r2, k_yaw=1.0):
  """ Like quat tilt priority but with rotation vectors
      Achieving similar to Mueller's 4th controller..."""
  Re = roterr(r1.as_matrix(), r2.as_matrix())
  dotp = np.clip(e3.T.dot(Re.dot(e3)), -1.0, 1.0)
  ang_reduced = np.arccos(dotp)

  sinang = np.sin(ang_reduced)
  if abs(sinang) < 1e-8:
    ax_reduced = e1
  else:
    ax_reduced = skew_matrix(Re.T.dot(e3)).dot(e3) / np.sin(ang_reduced)

  tv_part = 1.0 * ax_reduced * ang_reduced

  #ang_e = np.linalg.norm(R.from_matrix(Re).as_rotvec())
  #ang_yaw = 2 * np.arccos(np.cos(ang_e / 2.0) / np.cos(ang_reduced / 2.0))
  #ax_yaw = e3

  Ryaw = Re.dot(R.from_rotvec(ang_reduced * ax_reduced).inv().as_matrix())
  rv_yaw = R.from_matrix(Ryaw).as_rotvec()

  return tv_part + k_yaw * rv_yaw

def quat_tilt_priority(r1, r2, k_yaw=0.5):
  """ As in 2020 Brescianini's Tilt-prioritized Quadrocopter Attitude Control
      k_yaw should be between 0 and 1 """
  # TODO This fails very strangely for some reason.
  qerr = (r2 * r1.inv()).as_quat()
  #qerr = (r1 * r2.inv()).as_quat()
  #qerr = (r2.inv() * r1).as_quat()

  qfact = 1 / np.sqrt(qerr[3] ** 2 + qerr[2] ** 2)
  qerr_reduced = qfact * np.array((qerr[3] * qerr[0] - qerr[1] * qerr[2],
                                   qerr[3] * qerr[1] + qerr[0] * qerr[2],
                                   0))

  #Rred = R.from_quat(np.hstack((qerr_reduced, 1.0 / qfact)))
  #print((r2 * r1.inv()).apply(e3))
  #print(Rred.apply(e3))
  #input()

  #qerr_reduced = r1.inv().apply(qerr_reduced)

  qerr_yaw = qfact * np.array((0, 0, qerr[2]))
  k_reduced = 1.0

  return -k_reduced * qerr_reduced - 0 * k_yaw * np.sign(qerr[3]) * qerr_yaw

def quat_tilt_priority2(r1, r2, k_yaw=1.0):
  """ Controller (3) in Mueller's 2018 paper: QTP
      Seems to work whereas the QTP in the 2020 Brescianini paper behaves strangely """
  Re = roterr(r1.as_matrix(), r2.as_matrix())
  ang_reduced = np.arccos(e3.T.dot(Re.dot(e3)))
  ax_reduced = skew_matrix(Re.T.dot(e3)).dot(e3) / np.sin(ang_reduced)

  #Re2 = r2 * r1.inv()
  #qerr = Re2.as_quat()
  #qfact2 = qerr[3] ** 2 + qerr[2] ** 2
  #qfact = 1 / np.sqrt(qfact2)
  #qerr_reduced = qfact * np.array((qerr[3] * qerr[0] - qerr[1] * qerr[2],
  #                                 qerr[3] * qerr[1] + qerr[0] * qerr[2],
  #                                 0))

  #R_qerr = R.from_quat(np.hstack((qerr_reduced, qfact * qfact2)))

  #print(ax_reduced * np.sin(ang_reduced / 2))
  #print(-qerr_reduced)
  #input()

  #ang_e = np.linalg.norm(R.from_matrix(Re).as_rotvec())
  #ang_yaw = 2 * np.arccos(np.cos(ang_e / 2.0) / np.cos(ang_reduced / 2.0))
  #ax_yaw = e3

  Ryaw = Re.dot(R.from_rotvec(ang_reduced * ax_reduced).inv().as_matrix())
  rv_yaw = R.from_matrix(Ryaw).as_rotvec()
  ang_yaw = np.linalg.norm(rv_yaw)
  if ang_yaw > 1e-7:
    ax_yaw = rv_yaw / ang_yaw
  else:
    ax_yaw = e3

  return 2 * 1.0 * ax_reduced * np.sin(ang_reduced / 2) + 2 * k_yaw * ax_yaw * np.sin(ang_yaw / 2)

def tv_cross(r1, r2):
  #euler1 = r1.as_euler('ZYX')
  #euler2 = r2.as_euler('ZYX')
  #yawerr = normang(euler1[0])
  ## TODO The above yaw error does not seem right, needs to be angle around body z   # that needs to be travelled to meet desired yaw angle.

  ## r1.inv().apply(np.cross(r2m[:, 2], r1m[:, 2]))
  #aa = np.cross(r1.inv().apply(r2.apply(e3)), e3)
  #aa[2] = yawerr
  #return aa

  Re = roterr(r1.as_matrix(), r2.as_matrix())
  ang_reduced = np.arccos(e3.T.dot(Re.dot(e3)))
  ax_reduced = skew_matrix(Re.T.dot(e3)).dot(e3) / np.sin(ang_reduced)

  #Re2 = r2 * r1.inv()
  #qerr = Re2.as_quat()
  #qfact2 = qerr[3] ** 2 + qerr[2] ** 2
  #qfact = 1 / np.sqrt(qfact2)
  #qerr_reduced = qfact * np.array((qerr[3] * qerr[0] - qerr[1] * qerr[2],
  #                                 qerr[3] * qerr[1] + qerr[0] * qerr[2],
  #                                 0))

  #R_qerr = R.from_quat(np.hstack((qerr_reduced, qfact * qfact2)))

  #print(ax_reduced * np.sin(ang_reduced / 2))
  #print(-qerr_reduced)
  #input()

  #ang_e = np.linalg.norm(R.from_matrix(Re).as_rotvec())
  #ang_yaw = 2 * np.arccos(np.cos(ang_e / 2.0) / np.cos(ang_reduced / 2.0))
  #ax_yaw = e3

  Ryaw = Re.dot(R.from_rotvec(ang_reduced * ax_reduced).inv().as_matrix())
  rv_yaw = R.from_matrix(Ryaw).as_rotvec()
  ang_yaw = np.linalg.norm(rv_yaw)
  if ang_yaw > 1e-7:
    ax_yaw = rv_yaw / ang_yaw
  else:
    ax_yaw = e3

  return ax_reduced * np.sin(ang_reduced) + ax_yaw * np.sin(ang_yaw)

if __name__ == "__main__":
  thetas = np.linspace(0, np.pi - 0.01, 51)

  funcs = [
    (euler_zyx, "Euler"),
    (so3_discon, "SO(3) discon"),
    (so3_con, "SO(3) con"),
    (quat, "Quaternion"),
    (rotvec, "Rotation Vector"),
    #(rotvec_tilt_priority, "Rotation Vector: TP"),
    (rotvec_tilt_priority2, "Rotation Vector: TP2"),
    (quat_tilt_priority, "Quat: TP"),
    #(quat_tilt_priority2, "Quat: TP2"),
    #(tv_cross, "Thrust Vector Cross"),
  ]

  results = [[] for _ in funcs]

  r1 = R.from_euler('ZYX', [0.0, 0.0, 0.0])
  for theta in thetas:
    r2 = R.from_euler('ZYX', [0.0, 0.0, theta])

    for res, (f, _) in zip(results, funcs):
      res.append(-f(r1, r2)[0])

  named("Attitude Error Metrics")
  plt.xlabel("$\\theta$ (rad)")
  plt.ylabel("Error")

  for res, (_, s) in zip(results, funcs):
    plt.plot(thetas, res, label=s, linewidth=3)

  plt.plot(thetas, np.sin(thetas), '--', label="$\sin(\\theta)$", linewidth=3)

  plt.legend()
  plt.show()
