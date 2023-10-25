import numpy as np

from scipy.spatial.transform import Rotation as R

import quadsim.rot_metrics as rot_metrics

from python_utils.mathu import normang, e1, e2, e3

from quadsim.control import Controller, ControllerLearnAccel, torque_from_aa
from quadsim.flatness import (
  get_xdot_xddot,
  a_from_z,
  j_from_zdot,
  yaw_zyx_from_x,
  yawdot_zyx_from_xdot,
  alpha_from_zddot,
  alpha_from_flat,
  uddot_from_s,
  zddot_from_s,
  omega_from_zdot,
  omega_from_flat,
)

class FBLinController(Controller):
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

    # Linear control in flat snap space.
    acc = a_from_z(z, self.u)
    jerk = j_from_zdot(z, self.u, self.udot, zdot)

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)

    # Linear control in flat yaw space.
    yaw = yaw_zyx_from_x(x)
    yawdot = yawdot_zyx_from_xdot(x, xdot)

    yaw_error = normang(yaw - self.ref.yaw(t))
    yawdot_error = yawdot - self.ref.yawvel(t)

    yawacc = -self.Kyaw * yaw_error - self.Kyawvel * yawdot_error + self.ref.yawacc(t)

    angaccel_world = alpha_from_flat(self.u, acc, jerk, snap, yaw, yawdot, yawacc)

    # Needed?
    uddot = uddot_from_s(self.u, snap, z, zdot)

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    bodyz_force = self.model.mass * self.u
    torque = torque_from_aa(angaccel, self.model.I, state.ang)

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddot * self.dt

    self.vars.update(uddot=uddot)

    return self.out(bodyz_force, torque)

class FBLinControllerLearnAccel(ControllerLearnAccel):
  def __init__(self, model, learner, features, dt):
    self.Kpos = 6 * 120 * np.eye(3)
    self.Kvel = 4 * 120 * np.eye(3)
    self.Kacc = 120 * np.eye(3)
    self.Kjerk = 16 * np.eye(3)

    self.Kyaw = 30
    self.Kyawvel = 10

    self.gvec = np.array((0, 0, -model.g))

    self.model = model
    self.dt = dt

    self.reset()
    super().__init__(model, learner, features)

  def reset(self):
    self.u = self.model.g
    self.udot = 0

  def endtrial(self):
    self.reset()
    super().endtrial()

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)

    control_for_learner = None
    acc_error = self.accel_learner.testpoint(t, state, control_for_learner)
    dpos = self.accel_learner.dpos(t, state, control_for_learner)
    dvel = self.accel_learner.dvel(t, state, control_for_learner)

    acc = self.u * z + self.gvec + acc_error
    aed1 = dpos.dot(state.vel) + dvel.dot(acc)

    jerk = self.udot * z + self.u * zdot + aed1
    aed2 = dpos.dot(acc) + dvel.dot(jerk)

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)
    snap -= aed2

    uddot = snap.dot(z) + self.u * zdot.dot(zdot)
    zddot = (1.0 / self.u) * (snap - 2 * self.udot * zdot - uddot * z)
    angaccel_world = np.cross(z, zddot - np.cross(ang_world, zdot))

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    yaw = np.arctan2(x[1], x[0])
    yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / (x[0] ** 2 + x[1] ** 2)
    yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

    _, xddot = get_xdot_xddot(yawvel, yawacc, state.rot, zdot, zddot)
    alpha_cross_x = xddot - np.cross(ang_world, xdot)
    # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
    angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.u
    torque = self.model.I.dot(angaccel) + np.cross(state.ang, self.model.I.dot(state.ang))

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddot * self.dt

    self.add_datapoint(t, state, (bodyz_force, torque))

    return self.out(bodyz_force, torque)

class FBLinControllerThrustDelay(Controller):
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
    self.udes = model.g

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)

    udot = -self.model.motor_tc * (self.u - self.udes)

    acc = self.u * z + self.gvec
    jerk = udot * z + self.u * zdot

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)

    uddot = snap.dot(z) + self.u * zdot.dot(zdot)
    udesdot = (1.0 / self.model.motor_tc) * uddot + udot

    zddot = (1.0 / self.u) * (snap - 2 * udot * zdot - uddot * z)
    angaccel_world = np.cross(z, zddot - np.cross(ang_world, zdot))

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    yaw = np.arctan2(x[1], x[0])

    x_xy_norm = x[0] ** 2 + x[1] ** 2

    # Perhaps this is too limiting.
    # Should still include fblin terms in this case
    # should only turn off "yaw feedback".
    if x_xy_norm > 1e-8:
      yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / x_xy_norm
      yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

      _, xddot = get_xdot_xddot(yawvel, yawacc, state.rot, zdot, zddot)
      alpha_cross_x = xddot - np.cross(ang_world, xdot)
      # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
      angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.udes
    torque = torque_from_aa(angaccel, self.model.I, state.ang)

    # This assumes this controller is only called once every dt
    self.u += udot * self.dt
    self.udes += udesdot * self.dt

    self.vars.update(uddot=uddot)

    return self.out(bodyz_force, torque)

class FBLinControllerThrustDelayNL(Controller):
  """ Passes rotor delay through nonlinear model to get thrust. """
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
    self.udes = model.g
    self.torque_out = np.zeros(3)

    self.motormodeld = np.polyder(self.model.motor_thrust)
    self.motormodeldd = np.polyder(self.motormodeld)

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)

    # Assume torque is instant for now... alpha = alphades.
    rpmdes = self.model.rpm_from_forcetorque(self.model.mass * self.udes, self.torque_out)
    rpm = self.model.rpm_from_forcetorque(self.model.mass * self.u, self.torque_out)

    # Partial motor model w.r.t. RPMs
    fr_at_r = np.diag(self.motormodeld(rpm))

    term1 = self.model.motor_tc * self.model.mixer.dot(fr_at_r)
    udot = -term1.dot(rpm - rpmdes)[0] / self.model.mass

    acc = self.u * z + self.gvec
    jerk = udot * z + self.u * zdot

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)

    uddot = snap.dot(z) + self.u * zdot.dot(zdot)

    # if the motormodel is quadratic, this should not depend on rpm.
    frr_at_r = np.diag(self.motormodeldd(rpm))
    fr_at_rdes = np.diag(self.motormodeld(rpmdes))

    # rpm dot
    rdot = -self.model.motor_tc * (rpm - rpmdes)

    # TODO Is the below okay?
    alphadd = np.zeros(3)

    # TODO Include om cross I om term below.
    vdd = np.hstack((self.model.mass * uddot, self.model.I.dot(alphadd)))
    rdesdot = np.linalg.solve(term1, vdd + self.model.motor_tc * self.model.mixer.dot(frr_at_r).dot(rdot).dot(np.diag(rpm - rpmdes)) + self.model.motor_tc * self.model.mixer.dot(fr_at_r).dot(rdot))
    vdesdot = self.model.mixer.dot(fr_at_rdes).dot(rdesdot)
    udesdot = vdesdot[0] / self.model.mass
    #udesdot = (1.0 / self.model.motor_tc) * uddot + udot

    zddot = (1.0 / self.u) * (snap - 2 * udot * zdot - uddot * z)
    angaccel_world = np.cross(z, zddot - np.cross(ang_world, zdot))

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    yaw = np.arctan2(x[1], x[0])

    x_xy_norm = x[0] ** 2 + x[1] ** 2

    # Perhaps this is too limiting.
    # Should still include fblin terms in this case
    # should only turn off "yaw feedback".
    if x_xy_norm > 1e-8:
      yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / x_xy_norm
      yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

      _, xddot = get_xdot_xddot(yawvel, yawacc, state.rot, zdot, zddot)
      alpha_cross_x = xddot - np.cross(ang_world, xdot)
      # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
      angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.udes
    torque = torque_from_aa(angaccel, self.model.I, state.ang)
    self.torque_out = torque

    # This assumes this controller is only called once every dt
    self.u += udot * self.dt
    self.udes += udesdot * self.dt

    self.vars.update(uddot=uddot)

    return self.out(bodyz_force, torque)

class FBLinControllerFullDelay(Controller):
  """ Passes rotor delay through nonlinear model to get thrust. """
  def __init__(self, model, dt):
    super().__init__()

    #self.Kpos = 6 * 120 * np.eye(3)
    #self.Kvel = 4 * 120 * np.eye(3)
    #self.Kacc = 120 * np.eye(3)
    #self.Kjerk = 16 * np.eye(3)

    # Feedback is crackle... oh jeez
    #self.Kpos = 15120 * np.eye(3) # 126 x 5!
    #self.Kvel = 8400 * np.eye(3) # 70 x 5!
    #self.Kacc = 2100 * np.eye(3) # 17.5 x 5!
    #self.Kjerk = 300 * np.eye(3) # 2.5 x 5!
    #self.Ksnap = 25 * np.eye(3) # -0.208333 x 5!
    self.Kpos = 21600 * np.eye(3)
    self.Kvel = 14400 * np.eye(3)
    self.Kacc = 3600 * np.eye(3)
    self.Kjerk = 480 * np.eye(3)
    self.Ksnap = 30 * np.eye(3)

    self.Kyaw = 30
    self.Kyawvel = 10

    self.gvec = np.array((0, 0, -model.g))

    self.model = model
    self.dt = dt

    self.u = model.g
    self.udes = model.g
    self.udesdot = 0
    # This is in the world frame.
    self.alpha = np.zeros(3)
    # This is in the body frame.
    self.torque_out = np.zeros(3)

    self.motormodeld = np.polyder(self.model.motor_thrust)
    self.motormodeldd = np.polyder(self.motormodeld)

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)
    zddot = np.cross(self.alpha, z) + np.cross(ang_world, zdot)

    # Assume torque is instant for now... alpha = alphades.
    actual_torque = torque_from_aa(state.rot.inv().apply(self.alpha), self.model.I, state.ang)
    rpmdes = self.model.rpm_from_forcetorque(self.model.mass * self.udes, self.torque_out)
    rpm = self.model.rpm_from_forcetorque(self.model.mass * self.u, actual_torque)

    # Partial motor model w.r.t. RPMs
    fr_at_r = np.diag(self.motormodeld(rpm))

    term1 = self.model.motor_tc * self.model.mixer.dot(fr_at_r)
    udot = -term1.dot(rpm - rpmdes)[0] / self.model.mass

    # TODO Compute uddot using the nonlinear motor model as udot above.

    #udot  = -self.model.motor_tc * (self.u - self.udes)
    uddot = -self.model.motor_tc * (udot - self.udesdot)

    acc = self.u * z + self.gvec
    jerk = udot * z + self.u * zdot
    snap = uddot * z + 2 * udot * zdot + self.u * zddot

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)
    snap_error = snap - self.ref.snap(t)

    # For now assume that crackle_des is zero... hmm need 9th order traj
    crackle = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) - self.Ksnap.dot(snap_error)

    u3d = crackle.dot(z) + 3 * udot * zdot.dot(zdot) + 3 * self.u * zddot.dot(zdot)
    z3d = (1.0 / self.u) * (crackle - u3d * z - 3 * uddot * zdot - 3 * udot * zddot)

    udesddot = u3d / self.model.motor_tc + uddot

    # if the motormodel is quadratic, this should not depend on rpm.
    #frr_at_r = np.diag(self.motormodeldd(rpm))
    #fr_at_rdes = np.diag(self.motormodeld(rpmdes))

    # rpm dot
    #rdot = -self.model.motor_tc * (rpm - rpmdes)

    # TODO Is the below okay?
    #alphadd = np.zeros(3)
    # TODO Include om cross I om term below.
    #vdd = np.hstack((self.model.mass * uddot, self.model.I.dot(alphadd)))
    #rdesdot = np.linalg.solve(term1, vdd + self.model.motor_tc * self.model.mixer.dot(frr_at_r).dot(rdot).dot(np.diag(rpm - rpmdes)) + self.model.motor_tc * self.model.mixer.dot(fr_at_r).dot(rdot))
    #vdesdot = self.model.mixer.dot(fr_at_rdes).dot(rdesdot)
    #udesdot = vdesdot[0] / self.model.mass
    #udesdot = (1.0 / self.model.motor_tc) * uddot + udot

    aadot_xy_world = np.cross(z, z3d) + 2 * self.alpha.dot(z) * zdot + zdot.dot(zdot) * ang_world + ang_world.dot(z) * zddot

    aades = aadot_xy_world / self.model.motor_tc + self.alpha

    # Convert to body frame.
    angaccel = state.rot.inv().apply(aades)

    yaw = np.arctan2(x[1], x[0])

    x_xy_norm = x[0] ** 2 + x[1] ** 2

    # Perhaps this is too limiting.
    # Should still include fblin terms in this case
    # should only turn off "yaw feedback".
    if x_xy_norm > 1e-8:
      yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / x_xy_norm
      yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

      _, xddot = get_xdot_xddot(yawvel, yawacc, state.rot, zdot, zddot)
      alpha_cross_x = xddot - np.cross(ang_world, xdot)
      # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
      angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.udes
    torque = torque_from_aa(angaccel, self.model.I, state.ang)
    self.torque_out = torque

    # This assumes this controller is only called once every dt
    self.u += udot * self.dt
    self.udes += self.udesdot * self.dt
    self.udesdot += udesddot * self.dt
    self.alpha += aadot_xy_world * self.dt

    self.vars.update(uddot=uddot)

    return self.out(bodyz_force, torque)
