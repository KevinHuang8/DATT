import numpy as np

from scipy.spatial.transform import Rotation as R

import quadsim.rot_metrics as rot_metrics

from python_utils.mathu import e1, e2, e3

from quadsim.cascaded_utils import thrust_project_z
from quadsim.control import Controller
from quadsim.flatness import angvel_hod_from_flat

class CascadedControllerAngvel(Controller):
  """ Outputs angular velocity as the control instead of torque """
  def __init__(self, model, rot_metric=rot_metrics.euler_zyx, u_f=thrust_project_z):
    super().__init__()
    self.Kpos = 7 * np.eye(3)
    self.Kpos[2, 2] = 6
    self.Kvel = 4 * np.eye(3)
    self.Krot = 7.5 * np.eye(3)

    # Different yaw gains
    self.Krot[2, 2] = 3

    self.gvec = np.array((0, 0, -model.g))

    self.rot_metric = rot_metric
    self.u_f = u_f

    self.model = model
    self.sim_angvel = True
    self.vars = dict()

  def set_gains(self, pos, vel, rot):
    self.Kpos = np.diag(pos)
    self.Kvel = np.diag(vel)
    self.Krot = np.diag(rot)

  def response(self, t, state):
    accref = self.ref.acc(t)
    jerkref = self.ref.jerk(t)
    yawref = self.ref.yaw(t)
    yawvelref = self.ref.yawvel(t)

    # Position Control PD
    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    accdes = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) + accref

    # Reference Conversion
    udes = self.u_f(accdes - self.gvec, state.rot)
    rotdes, angveldes_w = angvel_hod_from_flat(udes, accdes, jerkref, yawref, yawvelref)

    self.vars.update(rotdes=rotdes)

    # Desires should be in the *current* body frame for control. (FBLin vs FFLin?)
    # This seems to be introducing some feedback linearization instead of feedforward linearization hmm...
    angveldes_b = state.rot.inv().apply(angveldes_w)

    # Attitude Control P
    rot_error = self.rot_metric(state.rot, rotdes)
    angvel = -self.Krot.dot(rot_error) + angveldes_b

    bodyz_force = self.model.mass * udes

    return bodyz_force, angvel
