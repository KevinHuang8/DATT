import numpy as np

from scipy.spatial.transform import Rotation as R

import DATT.quadsim.rot_metrics as rot_metrics

from DATT.python_utils.mathu import e3

from DATT.quadsim.cascaded_utils import thrust_project_z
from DATT.quadsim.control import Controller, ControllerLearnAccel, torque_from_aa
from DATT.quadsim.flatness import att_hod_from_flat, get_xdot_xddot

class CascadedController(Controller):
  def __init__(self, model, rot_metric=rot_metrics.euler_zyx, u_f=thrust_project_z, add_angvel_linearizing_term=False):
    super().__init__()
    self.Kpos = 7 * np.eye(3)
    self.Kpos[2, 2] = 6
    self.Kvel = 4 * np.eye(3)
    self.Krot = 120 * np.eye(3)
    self.Kang = 16 * np.eye(3)

    # Different yaw gains
    self.Krot[2, 2] = 30
    self.Kang[2, 2] = 10

    self.gvec = np.array((0, 0, -model.g))

    self.rot_metric = rot_metric
    self.u_f = u_f

    self.model = model
    self.add_angvel_linearizing_term = add_angvel_linearizing_term

  def set_gains(self, pos, vel, rot, ang):
    self.Kpos = np.diag(pos)
    self.Kvel = np.diag(vel)
    self.Krot = np.diag(rot)
    self.Kang = np.diag(ang)

  def response(self, t, state):
    accref = self.ref.acc(t)
    jerkref = self.ref.jerk(t)
    snapref = self.ref.snap(t)
    yawref = self.ref.yaw(t)
    yawvelref = self.ref.yawvel(t)
    yawaccref = self.ref.yawacc(t)

    # Position Control PD
    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    accdes = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) + accref

    # Reference Conversion
    udes = self.u_f(accdes - self.gvec, state.rot)
    rotdes, angveldes_w, angaccdes_w = att_hod_from_flat(udes, accdes, jerkref, snapref, yawref, yawvelref, yawaccref)

    self.vars.update(rotdes=rotdes)

    # Desires should be in the *current* body frame for control. (FBLin vs FFLin?)
    # This seems to be introducing some feedback linearization instead of feedforward linearization hmm...
    angveldes_b = state.rot.inv().apply(angveldes_w)
    angaccdes_b = state.rot.inv().apply(angaccdes_w)

    self.vars.update(angvel=angveldes_b)

    # Attitude Control PD
    rot_error = self.rot_metric(state.rot, rotdes)
    ang_error = state.ang - angveldes_b
    angaccel = -self.Krot.dot(rot_error) - self.Kang.dot(ang_error) + angaccdes_b
    if self.add_angvel_linearizing_term:
      angaccel += -state.ang[2] * np.cross(e3, state.ang)

    bodyz_force = self.model.mass * udes
    torque = torque_from_aa(angaccel, self.model.I, state.ang)

    return self.out(bodyz_force, torque)

class CascadedControllerLearnAccel(ControllerLearnAccel):
  def __init__(self, model, learner, features, rot_metric=rot_metrics.euler_zyx, u_f=thrust_project_z):
    self.Kpos = 6 * np.eye(3)
    self.Kvel = 4 * np.eye(3)
    self.Krot = 120 * np.eye(3)
    self.Kang = 16 * np.eye(3)

    # Different yaw gains
    self.Krot[2, 2] = 30
    self.Kang[2, 2] = 10

    self.gvec = np.array((0, 0, -model.g))

    self.rot_metric = rot_metric
    self.u_f = u_f

    self.model = model

    super().__init__(model, learner, features)

  def response(self, t, state):
    posref  = self.ref.pos(t)
    velref  = self.ref.vel(t)
    accref  = self.ref.acc(t)
    jerkref = self.ref.jerk(t)
    snapref = self.ref.snap(t)

    # Position Control
    pos_error = state.pos - posref
    vel_error = state.vel - velref

    accel_des = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.gvec + accref

    # TODO Figure out what the form of control (u) is in the learner
    control_for_learner = accel_des

    # TODO for this to be true feedforward linearization, need to query model at desired state.
    acc_error = self.accel_learner.testpoint(t, state, control_for_learner)
    accel_des -= acc_error

    dpos = self.accel_learner.dpos(t, state, control_for_learner)
    dvel = self.accel_learner.dvel(t, state, control_for_learner)

    # TODO: do not assume linear model below.
    aed1 = dvel.dot(accref) + dpos.dot(velref)
    aed2 = dvel.dot(jerkref) + dpos.dot(accref)

    jerkdes = jerkref - aed1
    snapdes = snapref - aed2

    # Reference Conversion
    yawdes = self.ref.yaw(t)
    yawveldes = self.ref.yawvel(t)
    yawaccdes = self.ref.yawacc(t)

    z_b = accel_des / np.linalg.norm(accel_des)

    # ZYX Euler angles yaw
    c2 = np.array((-np.sin(yawdes), np.cos(yawdes), 0))
    x_b = np.cross(c2, z_b)
    x_b /= np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)

    # ZXY Euler angles yaw
    #c1 = np.array((np.cos(yawdes), np.sin(yawdes), 0))
    #y_b = np.cross(z_b, c1)
    #y_b /= np.linalg.norm(y_b)
    #x_b = np.cross(y_b, z_b)

    rot_des = R.from_matrix(np.column_stack((x_b, y_b, z_b)))

    u = self.u_f(accel_des, state.rot)
    udot = jerkdes.dot(z_b)
    zdotdes = (1 / u) * (jerkdes - udot * z_b)
    uddot = snapdes.dot(z_b) + u * zdotdes.dot(zdotdes)
    zddotdes = (1 / u) * (snapdes - 2 * udot * zdotdes - uddot * z_b)

    angveldesxy_w = np.cross(z_b, zdotdes)
    angaccdesxy_w = np.cross(z_b, zddotdes - np.cross(angveldesxy_w, zdotdes))

    xdot, xddot = get_xdot_xddot(yawveldes, yawaccdes, rot_des, zdotdes, zddotdes)

    # See notes titled "Angular Velocity for Yaw" for proof of the below line
    omega_z_bdes = xdot.dot(y_b)
    angveldesxy_bdes = rot_des.inv().apply(angveldesxy_w)
    angveldes_bdes = np.hstack((angveldesxy_bdes[0:2], omega_z_bdes))

    angveldes_w = rot_des.apply(angveldes_bdes)
    alpha_cross_x = xddot - np.cross(angveldes_w, xdot)

    # See notes as for above omega line.
    alpha_z_bdes = alpha_cross_x.dot(y_b)
    angaccdesxy_bdes = rot_des.inv().apply(angaccdesxy_w)
    angaccdes_bdes = np.hstack((angaccdesxy_bdes[0:2], alpha_z_bdes))

    # Desires should be in the *current* body frame for control. (FBLin vs FFLin?)
    # This seems to be introducing some feedback linearization instead of feedforward linearization hmm...
    angveldes_b = state.rot.inv().apply(angveldes_w)
    angaccdes_b = state.rot.inv().apply(rot_des.apply(angaccdes_bdes))

    # Attitude Control
    rot_error = self.rot_metric(state.rot, rot_des)
    angaccel = -self.Krot.dot(rot_error) - self.Kang.dot(state.ang - angveldes_b) + angaccdes_b

    bodyz_force = self.model.mass * u
    torque = torque_from_aa(angaccel, self.model.I, state.ang)

    self.add_datapoint(t, state, (bodyz_force, torque))

    return self.out(bodyz_force, torque)
