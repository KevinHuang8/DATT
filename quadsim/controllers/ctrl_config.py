import numpy as np
import yaml
import torch
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR, TEST_POLICY_DIR
from quadsim.learning.utils.adaptation_network import AdaptationNetwork
from stable_baselines3.common.env_util import make_vec_env

# importing MPPI
from quadrotor_torch import Quadrotor_torch
from param_torch import Param, Timer
import controller_torch
from pathlib import Path
import os


def select_policy_config_(policy_config):
    log_scale = False
    body_frame = False
    relative = False
    task = None
    policy_name = None
    config_filename = None
    e_dims = 0
    adaptive_policy_name = None
    u_struct = False
    adaptation_warmup = False

    DIR = 'PPO'
    RMA = 'RMA'

    if policy_config=="hover":
        task: DroneTask = DroneTask.HOVER
        policy_name =DIR + '/' + "hover_04k"
        # policy_name =DIR + '/' + "hover_second_order_0606"
        config_filename = "default_hover.py"
        # config_filename = "hover_latency_0606.py"
        body_frame = True
        relative = False
        log_scale =False
        u_struct = False
        

    if policy_config == "trajectory":
        task: DroneTask = DroneTask.TRAJFBFF
        # policy_name =DIR + '/' + "traj_fbff_h10_p1_3i"
        # policy_name =DIR + '/' + "traj_random_zigzag_curriculum"
        # policy_name =DIR + '/' + "ppo_base"
        # policy_name =DIR + '/' + "traj_mixed_reset_thresh_5000_body_frame_relative_fixed3_7500000_steps.zip"
        policy_name =DIR + '/' + "ppo-rl-zigzag_x_bf_rel.zip" # traj_mixed_reset_thresh_5000_body_frame_relative_fixed3_7500000_steps.zip
        config_filename = "trajectory_latency.py"
        body_frame = True
        relative = True
    
    if policy_config == "trajectory_2d":
        task: DroneTask = DroneTask.TRAJFBFF
        # policy_name =DIR + '/' + "ppo-rl-zigzag_xy_bf_rel.zip" # traj_mixed2D_reset_thresh_5000_body_frame_relative_fixed3.zip
        policy_name = DIR + '/' + 'traj_mixed2D_all_refs_diffaxis2_17500000_steps.zip'
        # policy_name = DIR + '/' + "traj_mixed2D_symlog_noisy.zip"
        config_filename = "trajectory_latency.py"
        body_frame = True
        relative = True
        log_scale = False
    
    if policy_config == "trajectory_2d_sym":
        task: DroneTask = DroneTask.TRAJFBFF
        # policy_name =DIR + '/' + "ppo-rl-zigzag_xy_bf_rel.zip" # traj_mixed2D_reset_thresh_5000_body_frame_relative_fixed3.zip
        policy_name = DIR + '/' + "traj_mixed2D_symlog_noisy.zip"
        config_filename = "trajectory_latency.py"
        body_frame = True
        relative = True
        log_scale = True
    
    if policy_config == "trajectory_3d":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name =DIR + '/' + "traj_mixed3D_zmax03.zip" # traj_mixed2D_reset_thresh_5000_body_frame_relative_fixed3.zip
        config_filename = "trajectory_latency.py"
        body_frame = True
        relative = True
        log_scale = False
    
    if policy_config == "trajectory_3d_noise":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name =DIR + '/' + "traj_mixed3D_zmax03_symlog2.zip" # traj_mixed2D_reset_thresh_5000_body_frame_relative_fixed3.zip
        config_filename = "trajectory_latency.py"
        body_frame = True
        relative = True
        log_scale = True
    
            
    if policy_config == "trajectory_2d_mass_adaptive":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name =DIR + '/' + "traj_mixed2D_mass_adaptive_symlog.zip"
        adaptive_policy_name =  RMA + '/' + 'RMA_mass_nowind.pth'
        config_filename = "trajectory_wind_adaptive.py"
        e_dims = 1
        body_frame = True
        relative = True
        log_scale = True
    
    if policy_config == "trajectory_2d_wind_adaptive":
        task: DroneTask = DroneTask.TRAJFBFF
        # policy_name =DIR + '/' + "traj_mixed2D_wind_adaptive2_REAL"
        # policy_name =DIR + '/' + "traj_mixed2D_wind_high_wind"
        policy_name = DIR + '/' + "traj_mixed2D_wind_L1_randomwalk8_22500000_steps.zip"
        # policy_name = DIR + '/' + "traj_mixed2D_wind_L1.zip"
        # policy_name = DIR + '/' + "traj_mixed2D_wind_randomwalk_low.zip"

        adaptive_policy_name =  RMA + '/' + 'wind_adaptation_net_RMA.pth'
        config_filename = "trajectory_wind_adaptive.py"
        e_dims = 3
        body_frame = True
        relative = True
        log_scale = True
    
    # if policy_config == "trajectory_2d_wind_adaptive":
    #     task: DroneTask = DroneTask.TRAJFBFF
    #     # policy_name =DIR + '/' + "traj_mixed2D_wind_adaptive2_REAL"
    #     # policy_name =DIR + '/' + "traj_mixed2D_wind_high_wind"
    #     policy_name = DIR + '/' + "traj_mixed2D_wind_L1_randomwalk8_22500000_steps.zip"
    #     # policy_name = DIR + '/' + "traj_mixed2D_wind_L1.zip"
    #     # policy_name = DIR + '/' + "traj_mixed2D_wind_randomwalk_low.zip"

    #     adaptive_policy_name =  RMA + '/' + 'wind_adaptation_net_RMA.pth'
    #     config_filename = "trajectory_wind_adaptive.py"
    #     e_dims = 3
    #     body_frame = True
    #     relative = True
    #     log_scale = True
    
    # if policy_config == "trajectory_2d_wind_adaptive_RMA":
    #     task: DroneTask = DroneTask.TRAJFBFF
    #     # policy_name =DIR + '/' + "traj_mixed2D_wind_adaptive2_REAL"
    #     # policy_name =DIR + '/' + "traj_mixed2D_wind_high_wind"
    #     policy_name = DIR + '/' + "traj_mixed2D_wind_L1_randomwalk8_22500000_steps.zip"
    #     # policy_name = DIR + '/' + "traj_mixed2D_wind_L1.zip"
    #     # policy_name = DIR + '/' + "traj_mixed2D_wind_randomwalk_low.zip"

    #     adaptive_policy_name =  RMA + '/' + 'wind_adaptation_net_RMA'
    #     config_filename = "trajectory_wind_adaptive.py"
    #     e_dims = 3
    #     body_frame = True
    #     relative = True
    #     log_scale = True
    
    if policy_config == "trajectory_2d_mass_wind_adaptive":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name =DIR + '/' + "traj_mixed2D_mass_wind_adaptive_low.zip"
        adaptive_policy_name =  RMA + '/' + 'mass_wind_noisy_adaptation_net_9000'
        config_filename = "trajectory_mass_wind_adaptive.py"
        e_dims = 4
        body_frame = True
        relative = True
        log_scale = False

    if policy_config == "trajectory_2d_mass_wind_adaptive_ustruct":
        task: DroneTask = DroneTask.TRAJFBFF
        # policy_name =DIR + '/' + "rwik_mixed2D_ustruct_0.4_12500000_steps.zip"
        policy_name = DIR + "/" + "rwik_mixed2D_ustruct_mass_wind_0.4_noisy_20000000_steps.zip"
        adaptive_policy_name =  RMA + '/' + 'rwik_adapt_2_4500'
        config_filename = "trajectory_mass_wind_adaptive.py"
        e_dims = 4
        body_frame = True
        relative = True
        log_scale = True
        u_struct = True
    
    if policy_config == "trajectory_2d_mass_wind_adaptive_noise":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name  = DIR + '/' + 'traj_mixed2D_mass_wind_adaptive_best.zip'
        adaptive_policy_name =  RMA + '/' + 'mass_wind_adaptation_net_noisy'
        config_filename = "trajectory_mass_wind_adaptive.py"
        e_dims = 4
        body_frame = True
        relative = True
        log_scale = False
    
    if policy_config == "trajectory_2d_wind_adaptive":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name  = DIR + '/' + 'traj_mixed2D_wind_adaptive2_REAL.zip'
        config_filename = "trajectory_wind_adaptive.py"
        e_dims = 3
        body_frame = True
        relative = True
        log_scale = False
    
    if policy_config == "trajectory_2d_wind_adaptive_RMA":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name  = DIR + '/' + 'traj_mixed2D_wind_adaptive2_REAL.zip'
        adaptive_policy_name =  RMA + '/' + 'wind_adaptation_net_RMA'
        config_filename = "trajectory_wind_adaptive.py"
        e_dims = 3
        body_frame = True
        relative = True
        log_scale = False
    
    if policy_config == "trajectory_2d_mass_wind_adaptive_noise_warmup":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name  = DIR + '/' + 'traj_mixed2D_mass_wind_adaptive_best.zip'
        adaptive_policy_name =  RMA + '/' + 'hi_mass_low_wind_adaptation_net_REAL'
        config_filename = "trajectory_mass_wind_adaptive.py"
        e_dims = 4
        body_frame = True
        relative = True
        log_scale = False
        adaptation_warmup = True
    
    if policy_config == "trajectory_2o":
        task: DroneTask = DroneTask.TRAJFBFF
        policy_name =DIR + '/' + "traj_mixed2D_2nd_order_0404"
        config_filename = "trajectory_latency_0404.py"
        body_frame = True
        relative = True
        log_scale = False
    
    if policy_config == "yawflip":
        task: DroneTask = DroneTask.YAWFLIP
        policy_name =DIR + '/' + "yawflip_high_latency_2.zip"
        config_filename = "yawflip_latency.py"
        body_frame = False
        relative = False
        log_scale = False
    
    policy_dict = {"task":task,
                   "policy_name":policy_name,
                   "config_filename":config_filename,
                   "adaptive_policy_name":adaptive_policy_name,
                   "body_frame":body_frame,
                   "relative":relative,
                   "log_scale":log_scale,
                   "e_dims":e_dims,
                   'u_struct':u_struct,
                   'adaptation_warmup':adaptation_warmup}
    
    return policy_dict
