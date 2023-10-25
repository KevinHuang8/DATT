import numpy as np

from DATT.python_utils.mathu import quat_identity, quat_mult, vector_quat, matrix_from_quat, rodmat

from scipy.spatial.transform import Rotation as R

def exp_quat(v):
  """ See "Practical Parameterization of Rotations Using the Exponential Map"
      - F. Sebastian Grassia
      Section 3
  """
  ang = np.linalg.norm(v)
  hang = 0.5 * ang
  if ang < 1e-8: # Threshold probably too conservative (too low)
    sc = 0.5 + ang ** 2 / 48
  else:
    sc = np.sin(hang) / ang

  return np.array((np.cos(hang), sc * v[0], sc * v[1], sc * v[2]))

def euler_int(dt, vel, accel):
  return vel * dt + 0.5 * accel * dt ** 2

def so3_quat_int(quat, dang, dang_in_body):
  """ Uses exponential map for quaternions """
  rotquat = exp_quat(dang)
  if dang_in_body:
    return quat_mult(quat, rotquat)

  return quat_mult(rotquat, quat)

def so3_quat_naive_int(quat, dang, dang_in_body):
  """ Uses quaternion derivative equation and normalizes """
  if dang_in_body:
    quat_deriv = quat_mult(quat, vector_quat(dang)) / 2.0
  else:
    quat_deriv = quat_mult(vector_quat(dang), quat) / 2.0

  quat += quat_deriv
  return quat / np.linalg.norm(quat)

def so3_rot_int(rot, dang, dang_in_body):
  """ Uses exponential map for rotation matrices """
  delta_rot = rodmat(dang)
  if dang_in_body:
    return rot.dot(delta_rot)

  return delta_rot.dot(rot)

class AttitudeNoLie(object):
  def __init__(self, quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.quat = quat.copy()
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    dang = euler_int(dt, self.ang, ang_accel)
    self.quat = so3_quat_naive_int(self.quat, dang, self.in_body)
    self.ang += ang_accel * dt

  def get_quat(self):
    return self.quat.copy()

  def get_rot(self):
    return matrix_from_quat(self.quat)

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    quat = R.from_matrix(rot).as_quat()
    self.quat = np.array((quat[3], quat[0], quat[1], quat[2]))

class Attitude(object):
  def __init__(self, quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.quat = quat.copy()
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    dang = euler_int(dt, self.ang, ang_accel)
    self.quat = so3_quat_int(self.quat, dang, self.in_body)
    self.ang += ang_accel * dt

  def get_quat(self):
    return self.quat.copy()

  def get_rot(self):
    return matrix_from_quat(self.quat)

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    quat = R.from_matrix(rot).as_quat()
    self.quat = np.array((quat[3], quat[0], quat[1], quat[2]))

class AttitudeRot(object):
  def __init__(self, quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.rot = matrix_from_quat(quat)
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    dang = euler_int(dt, self.ang, ang_accel)
    self.rot = so3_rot_int(self.rot, dang, self.in_body)
    self.ang += ang_accel * dt

  def get_rot(self):
    return self.rot.copy()

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    self.rot = rot

class RigidBody3D(object):
  """
    quat transforms body to world.
    ang and ang_accel are in the world frame and radians (unless in_body=True).
    vel and accel are in the world frame and meters.

    Uses exponential map for quaternions.
  """
  def __init__(self, pos=np.zeros(3), vel=np.zeros(3), quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.pos = pos.copy()
    self.vel = vel.copy()
    self.quat = quat.copy()
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    self.pos += euler_int(dt, self.vel, accel)
    self.vel += accel * dt

    dang = euler_int(dt, self.ang, ang_accel)
    self.quat = so3_quat_int(self.quat, dang, self.in_body)
    self.ang += ang_accel * dt

  def get_pos(self):
    return self.pos.copy()

  def get_vel(self):
    return self.vel.copy()

  def get_quat(self):
    return self.quat.copy()

  def get_rot(self):
    return matrix_from_quat(self.quat)

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    quat = R.from_matrix(rot).as_quat()
    self.quat = np.array((quat[3], quat[0], quat[1], quat[2]))

class RigidBody3DRot(object):
  """
    rot transforms body to world.
    ang and ang_accel are in the world frame and radians (unless in_body=True).
    vel and accel are in the world frame and meters.

    Use Exponential Map for rotation matrices (SO(3)).
  """
  def __init__(self, pos=np.zeros(3), vel=np.zeros(3), quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.pos = pos.copy()
    self.vel = vel.copy()
    self.rot = matrix_from_quat(quat)
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    self.pos += euler_int(dt, self.vel, accel)
    self.vel += accel * dt

    dang = euler_int(dt, self.ang, ang_accel)
    self.rot = so3_rot_int(self.rot, dang, self.in_body)
    self.ang += ang_accel * dt

  def get_pos(self):
    return self.pos.copy()

  def get_vel(self):
    return self.vel.copy()

  def get_rot(self):
    return self.rot.copy()

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    self.rot = rot

if __name__ == "__main__":
  import time

  ang0 = 0 * np.array((1.0, 0, 0))

  body_rot = AttitudeRot(ang=ang0)
  body_quat = Attitude(ang=ang0)
  body_quat_nolie = AttitudeNoLie(ang=ang0)

  rot_ts = []
  quat_ts = []
  quatnolie_ts = []

  N = 2000
  dt = 0.02
  acc_mag = 2 * np.pi

  aa_base = np.random.normal(size=(3,))
  aa_base /= np.linalg.norm(aa_base)

  for i in range(N):
    if i < N / 2:
      ang_acc = acc_mag
    else:
      ang_acc = -acc_mag

    aa = ang_acc * aa_base

    t1 = time.process_time()
    body_rot.step(dt, ang_accel=aa)
    t2 = time.process_time()
    body_quat.step(dt, ang_accel=aa)
    t3 = time.process_time()
    body_quat_nolie.step(dt, ang_accel=aa)
    t4 = time.process_time()

    rot_ts.append(t2 - t1)
    quat_ts.append(t3 - t2)
    quatnolie_ts.append(t4 - t3)

  res2 = matrix_from_quat(body_quat.quat)
  res3 = body_rot.rot

  data = [
    ("Quat + Norm", matrix_from_quat(body_quat_nolie.quat), quatnolie_ts),
    ("Quat w/ Exp. Map", matrix_from_quat(body_quat.quat), quat_ts),
    ("SO(3) w/ Exp. Map", body_rot.rot, rot_ts)
  ]
  print("N = %d, dt = %f" % (N, dt))
  for s, rot, _ in data:
    print("%s (%s)" % (s, ["WRONG - did not return to origin", "CORRECT"][np.allclose(rot, np.eye(3))]))
    print('\tEuler angles: %s' % str(R.from_matrix(rot).as_euler('ZYX')))

  print("Avg time per step (us)")
  for s, _, ts in data:
    print("\t%s: %f (%f std)" % (s, 1e6 * np.mean(ts), 1e6 * np.std(ts)))
