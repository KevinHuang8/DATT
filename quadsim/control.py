import numpy as np

from DATT.quadsim.learn import AccelLearner

class Controller:
  def __init__(self, output_rpm=False):
    self.output_rpm = output_rpm
    self.vars = {}

    self.sim_angvel = False

  def endtrial(self):
    pass

  def out(self, force, torque):
    if self.output_rpm:
      # This is quite hacky.
      self.vars.update(forcedes=force, torquedes=torque)
      return self.model.rpm_from_forcetorque(force, torque)

    return force, torque

class ControllerLearnAccel(Controller):
  def __init__(self, model, learner, features):
    super().__init__()
    self.accel_learner = AccelLearner(model, learner, features)

  def endtrial(self):
    self.accel_learner.train()
    self.accel_learner.reset_trial()

  def add_datapoint(self, t, state, control):
    accel_error_true = self.accel_learner.add(t, state, control)
    accel_error_pred = self.accel_learner.testpoint(t, state, control)
    self.vars.update(accel_error_true=accel_error_true, accel_error_pred=accel_error_pred)

def torque_from_aa(aa, I, ang):
  """ Returns torque given ang. accel. (and inertia and ang. vel.). """
  return I.dot(aa) + np.cross(ang, I.dot(ang))
