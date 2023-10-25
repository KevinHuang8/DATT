import numpy as np

from python_utils.mathu import e3

class InputPos:
  dim = 3
  def get(self, t, state, control):
    return state.pos

  def dpos(self, t, state, control):
    return np.eye(3)

  def dvel(self, t, state, control):
    return np.zeros((3, 3))

class InputVel:
  dim = 3
  def get(self, t, state, control):
    return state.vel

  def dpos(self, t, state, control):
    return np.zeros((3, 3))

  def dvel(self, t, state, control):
    return np.eye(3)

class InputPosVel:
  dim = 6
  def get(self, t, state, control):
    return np.hstack((state.pos, state.vel))

  def dpos(self, t, state, control):
    ret = np.zeros((6, 3))
    ret[:3, :3] = np.eye(3)
    return ret

  def dvel(self, t, state, control):
    ret = np.zeros((6, 3))
    ret[3:, :3] = np.eye(3)
    return ret

class AccelLearner:
  def __init__(self, model, learner, input_cls):
    self.lastt = 0
    self.lastvel = np.zeros(3)
    self.last_acc = model.g * e3
    self.gvec = -model.g * e3
    self.mass = model.mass

    self.learner = learner
    self.input_cls = input_cls

    self.reset_trial()
    self.reset_data()

  def reset_data(self):
    self.xs = []
    self.ys = []
    self.trained = False

  def reset_trial(self):
    self.first = True

  def add(self, t, state, control):
    dt = t - self.lastt
    self.lastt = t

    dvel = state.vel - self.lastvel
    self.lastvel = state.vel

    accel_base = self.last_acc + self.gvec
    z = state.rot.apply(e3)
    self.last_acc = control[0] * z / self.mass

    if self.first:
      self.first = False
      return np.zeros(3)

    accel_true = dvel / dt
    accel_err = accel_true - accel_base

    self.xs.append(self.get_input(t, state, control))
    self.ys.append(accel_err)

    return accel_err

  def train(self):
    self.trained = True
    self.learner.train(np.array(self.xs), np.array(self.ys))

  def get_input(self, t, state, control):
    return self.input_cls.get(t, state, control)

  def testpoint(self, t, state, control):
    if not self.trained:
      return np.zeros(3)

    xs = np.array([self.get_input(t, state, control)])
    return self.learner.test(xs)[0]

  def generic_d(self, dname, t, state, control):
    if not self.trained:
      return np.zeros(3)

    learner_gradient = self.learner.gradient(self.get_input(t, state, control))
    features_gradient = getattr(self.input_cls, dname)(t, state, control)
    return learner_gradient.dot(features_gradient)

  def dpos(self, t, state, control):
    return self.generic_d('dpos', t, state, control)
  def dvel(self, t, state, control):
    return self.generic_d('dvel', t, state, control)
