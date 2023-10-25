# quadsim

## Setup

First, install dependencies.

Make sure you have numpy, scipy, and matplotlib (for Python 3).
(Can install these using your system's package manager).

Install meshcat e.g. with `pip install meshcat` (required for visualization).

Learning policies using RL requires `gym` and `stable_baselines3`.
Run `pip install gym stable-baselines3[extra]`.
If you run into an error about protoc and buffers being out of date, try running `pip install tensorboard -U`.

Quadsim requires the `python_utils` repo to exist on a `PYTHONPATH`.

The recommended setup is to create a folder named "python" (e.g. in your home folder) and then clone both `python_utils` and `quadsim` in `~/python`.

```
mkdir ~/python
cd ~/python
git clone https://github.com/alspitz/python_utils
git clone https://github.com/KevinHuang8/DATT
```

Next, in your `.bashrc`, add `${HOME}/python` to `PYTHONPATH`.
e.g. add the following line.
```
export PYTHONPATH="${HOME}/python":"${PYTHONPATH}"
```

Change directory and rc file as needed (e.g. if using zsh).

## Usage

Run `python interactive.py`, `python main.py`, `python compare.py`, or `python learntest.py`.

A browser window should open with a visualization of the quadrotor executing the reference trajectories with the specified controllers.

After that, plots should appear showing state and control variables.

[interactive.py](interactive.py) allows you to send step commands to the robot using WASD, QE, and -+ (while focused on the terminal).

[compare.py](compare.py) has some disturbances and additional settings that can be enabled at the bottom of the file.

[learntest.py](learntest.py) iteratively learns a dynamics model to mitigate the effects of a static wind field.

See below for what the meshcat visualization should look like.
![Meshcat visualization](media/meshcat-cf.png)
