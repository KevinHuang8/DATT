import numpy as np

from scipy.spatial.transform import Rotation as R

import quadsim.rot_metrics as rot_metrics

from python_utils.mathu import normang, e1, e2, e3

from quadsim.control import Controller, ControllerLearnAccel, torque_from_aa
from quadsim.flatness import (
  u1_from_a,
  a_from_z,
  j_from_zdot,
  yaw_zyx_from_x,
  yawdot_zyx_from_xdot,
  uddot_from_flat,
  alpha_from_flat,
)

class FFLinController(Controller):
  def __init__(self, model, dt):
    super().__init__()

    self.Kpos = 6 * 120 * np.eye(3)
    self.Kvel = 4 * 120 * np.eye(3)
    self.Kacc = 120 * np.eye(3)
    self.Kjerk = 16 * np.eye(3)

    self.Kyaw = 30
    self.Kyawvel = 10

    self.gvec = np.array((0, 0, -model.g))

    self.model = model
    self.dt = dt

    self.u = model.g
    self.udot = 0

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)

    accref = self.ref.acc(t)
    jerkref = self.ref.jerk(t)
    snapref = self.ref.snap(t)

    yawref = self.ref.yaw(t)
    yawvelref = self.ref.yawvel(t)
    yawaccref = self.ref.yawacc(t)

    # Linear control in flat snap space.
    acc = a_from_z(z, self.u)
    jerk = j_from_zdot(z, self.u, self.udot, zdot)

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - accref
    jerk_error = jerk - jerkref

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + snapref

    uref = u1_from_a(accref) # This or self.u?

    # Linear control in flat yaw space.
    yaw = yaw_zyx_from_x(x)
    yawdot = yawdot_zyx_from_xdot(x, xdot)

    yaw_error = normang(yaw - self.ref.yaw(t))
    yawdot_error = yawdot - self.ref.yawvel(t)

    yawaccdes = -self.Kyaw * yaw_error - self.Kyawvel * yawdot_error + self.ref.yawacc(t)

    # Invert at reference states? Is this feedforward linearization?
    angaccel_world = alpha_from_flat(uref, accref, jerkref, snap, yawref, yawvelref, yawaccdes)

    # Needed?
    uddotdes = uddot_from_flat(uref, accref, jerkref, snap)

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    bodyz_force = self.model.mass * self.u
    torque = torque_from_aa(angaccel, self.model.I, state.ang)

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddotdes * self.dt

    self.vars.update(uddot=uddotdes)

    return self.out(bodyz_force, torque)
