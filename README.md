# DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control

<div align="center">

[[Website]](https://sites.google.com/view/deep-adaptive-traj-tracking)
[[arXiv]](https://arxiv.org/abs/2310.09053)

[![Python Version](https://img.shields.io/badge/Python-3.8-blue.svg)](https://github.com/KevinHuang8/DATT)


![](images/main.png)
</div>

## Setup

First, install `requirements.txt`.

The repo requires the parent folder exist on `PYTHONPATH`.

The recommended setup is to create a folder named "python" (e.g. in your home folder) and then clone `DATT` in `~/python`.

```
mkdir ~/python
cd ~/python
git clone https://github.com/KevinHuang8/DATT
```

Next, in your `.bashrc`, add `${HOME}/python` to `PYTHONPATH`.
e.g. add the following line.
```
export PYTHONPATH="${HOME}/python":"${PYTHONPATH}"
```

Change directory and rc file as needed (e.g. if using zsh).

## Usage


### Summary:

For training the main policy from the DATT paper, from the `learning` folder, run:

`python train_policy.py -n policy -c DATT_config.py -t trajectory_fbff --ref mixed_ref -ts 25000000 --checkpoint True` 

To evaluate the policy, run:

`python eval_policy.py -n policy -c DATT_config.py -t trajectory_fbff --ref random_zigzag -s 500 --viz True`

More detailed instructions to come.

### Setting up a task / configuration

Training a policy requires specifying a *task* and a *configuration*. The task describes the environment and reward, while the configuration defines various environmental parameters, such as drone mass, wind, etc., and whether/how they are randomly sampled.

### Tasks

See tasks in `./tasks/`. Each class is superclass of `BaseEnv`, which has the gym env API. In practice, the primary thing that should change between different drone tasks are the action space and reward function.

Standard trajectory tracking should have `trajectory_fbff` passed in.

NOTE: When adding a new task, you must modify the `DroneTask` enum in `train_policy.py` to add the new task along with its corresponding environment, for it to get recognized. 

### Configuration

`./configuration/configuration.py` defines all the modifiable parameters, as well as their default values. To define a configuration, create a new `.py` file that instantiates a `AllConfig` object named `config`, which modifies the config values for parameters that are different from the default values. See config profiles in `./configuration/` for examples.

**NOTE: `configuration.py` should not be modified (it just defines the configurable parameters). To create a configuration, a new file needs to be created.**

Configurable parameters that can be randomly sampled during training can be set to a `ConfigValue` (see `configuration.py`). Each `ConfigValue` takes in the default value of the parameter, and whether or not that parameter should be randomized during training. If a param should be randomized, you need to also specify the min and max possible range of randomization for that parameter.

Each parameter is part of some parameter group, which shares a `Sampler`, which specifies how parameters in that group should be randomly sampled if they are specified to be randomized. By default, the sampling function is just uniform sampling, but the sampling function can take in anything, like the reward or timestep, which can be used to specify a learning curriculum, etc. To add more info to the sampling function input, or to change *when* in training a parameter is resampled from the default, however, you must modify the task/environment.

# Training a policy

Run `train_policy.py` from the command line. It takes the following arguments:

- `-n` `--name` : the name of the policy. All log/data files will have this name as the prefix. If you pass in a name that already exists (a policy exists if a file with the same name appears in `./saved_policies/`), then you will continue training that policy with new data. If not provided, autogenerates depending on the other parameters.
- `-t` `--task` : the name of the task; must be defined in the `DroneTask` enum in `train_policy.py`. Essentially, this specifies the environment. Defaults to hovering
- `-c` `--config` : The configuration file (a `.py` file), which must instantiate an `AllConfig` object named `config`. 
- `-ts` `--timesteps` : The number of timesteps to train for, defaults to 1 million. The model also saves checkpoints every 500,000 steps.
- `-d` `--log-dir` : The directory to save training logs (tensorboard) to. Defaults to `./logs/{policy_name}_logs`

**NOTE: must run `train_policy.py` from the `./learning/` directory for save directories to line up correctly.**

# Evaluating a policy

Run `eval_policy.py` with the policy name, task, algorithm the policy was trained on, and the number of evaluation steps.

This script currently just prints out the mean/std rewards over randomized episodes, and visualizes rollouts of the policy in sim.
