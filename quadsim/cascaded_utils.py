import numpy as np

from DATT.python_utils.mathu import e3

def thrust_project_z(accel_des, rot):
  z_b = rot.apply(e3)
  return accel_des.dot(z_b)

def thrust_norm_accel(accel_des, rot):
  return np.linalg.norm(accel_des)

def thrust_maintain_z(accel_des, rot):
  z_b = rot.apply(e3)
  return accel_des.dot(e3) / z_b.dot(e3)
