import numpy as np

from scipy.spatial.transform import Rotation as R
from DATT.quadsim.control import Controller
from DATT.quadsim.models import RBModel
from DATT.quadsim.controllers.cntrl_config import PIDConfig
from DATT.quadsim.rigid_body import State_struct

class PIDController(Controller):
  def __init__(self, model : RBModel, cntrl_config : PIDConfig):
    super().__init__()
    self.model = model
    self.kp_pos = 6
    self.kd_pos = 4
    self.ki_pos = 1.5
    self.kp_rot = 120
    self.kd_rot = 16

    self.prev_t = None
    self.pos_err_int = np.zeros(3)

  def response(self, t, state : State_struct):
    """
        Given a time t and state state, return body z force and torque (body-frame).

        State is defined in rigid_body.py and includes pos, vel, rot, and ang.

        The reference is available using self.ref (defined in flatref.py)
        and contains pos, vel, acc, jerk, snap, yaw, yawvel, and yawacc,
        which are all functions of time.

        self.model contains quadrotor model parameters such as mass, inertia, and gravity.
        See models.py.

        TODO Implement a basic quadrotor controller.
        E.g. you may want to compute position error using something like state.pos - self.ref.pos(t).

        You can test your controller by running main.py.
    """
    if self.prev_t is None:
      dt = 0
    else:
      dt = t - self.prev_t
    self.prev_t = t

    p_err = state.pos - self.ref.pos(t)
    v_err = state.vel - self.ref.vel(t)
    self.pos_err_int += p_err * dt 
    u_des = self.model.mass * (np.array([0, 0, self.model.g]) - self.kp_pos*(p_err) - self.kd_pos*(v_err) - self.ki_pos*self.pos_err_int + self.ref.acc(t))

    bodyz_force = np.linalg.norm(u_des)

    # Compute omega_des
    z_des = u_des / np.linalg.norm(u_des)
    bodyz_force_dot = self.ref.jerk(t).dot(z_des)
    z_des_dot = 1 / bodyz_force * (self.ref.jerk(t) - bodyz_force_dot * z_des)
    # omega_des in world frame, without yaw
    omega_des = np.cross(z_des, z_des_dot)
    # Convert to body frame
    omega_des = state.rot.as_matrix().T.dot(omega_des)

    # Compute alpha_des
    bodyz_force_2dot = self.ref.snap(t).dot(z_des) + bodyz_force*(z_des_dot.dot(z_des_dot))
    z_des_2dot = 1 / bodyz_force * (self.ref.snap(t) - bodyz_force_2dot*z_des - 2*bodyz_force_dot*z_des_dot)
    alpha_des = np.cross(z_des, z_des_2dot)
    # convert to body frame
    alpha_des = state.rot.as_matrix().T.dot(alpha_des)

    # Convert everything to body frame
    u_des = state.rot.as_matrix().T.dot(u_des)
    bodyz_force = np.linalg.norm(u_des)
    # print('pos', state.pos, 'pos_err', p_err, 'int', self.pos_err_int)

    rot_err = np.cross(u_des / bodyz_force, np.array([0, 0, 1]))
    ang_err = state.ang - omega_des
    alpha_fb = -self.kp_rot * rot_err - self.kd_rot * ang_err + alpha_des
    torque = self.model.I.dot(alpha_fb)

    yaw, _, _ = state.rot.as_euler('zyx')
    yaw_des = self.ref.yaw(t)
    yaw_err = yaw - yaw_des
    u_yaw = self.kp_rot * yaw_err - self.kd_rot * state.ang[2]
    # torque[2] = u_yaw

    return self.out(bodyz_force, torque)
