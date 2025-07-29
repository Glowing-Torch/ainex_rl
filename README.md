
# Installation
Our code is built on top of the repository: [Humanoid-Gym](https://github.com/roboterax/humanoid-gym).

1. Generate a new Python virtual environment with Python 3.8 using `conda create -n humanoid python=3.8`.
2. For the best performance, we recommend using NVIDIA driver version 525 `sudo apt install nvidia-driver-525`. The minimal driver version supported is 515. If you're unable to install version 525, ensure that your system has at least version 515 to maintain basic functionality.
3. Install PyTorch 1.13 with Cuda-11.7:
   - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. Install numpy-1.23 with `conda install numpy=1.23`.
5. Install Isaac Gym:
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - Run an example with `cd examples && python 1080_balls_of_solitude.py`.
   - Consult `isaacgym/docs/index.html` for troubleshooting.
6. Install ainex_rl:
   - Clone this repository.
   - `cd ainex_rl && pip install -e .`

# Usage Guide

## Examples
### 1. Train and Play
```bash
# Under the directory ainex_rl/humanoid
# Launching PPO Policy Training for 'v1' 
# This command initiates the PPO algorithm-based training for the humanoid task.
python humanoid/scripts/train.py --run_name v1 --headless 

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment. 
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python humanoid/scripts/play.py --load_run log_file_path --name run_name
# Run our trained policy
python humanoid/scripts/play.py --load_run Jul21_15-20-32_omniverse_edit_urdf --run_name omniverse_edit_urdf 
```

### 2. Sim-to-sim
- **Please note: Before initiating the sim-to-sim process, ensure that you run `play.py` to export a JIT policy.**
- **Mujoco-based Sim2Sim Deployment**: Utilize Mujoco for executing simulation-to-simulation (sim2sim) deployments with the command below:
```bash
python humanoid/scripts/sim2sim.py --load_model policy_1.pt
# Run our trained policy
python humanoid/scripts/sim2sim.py --load_model dt0.001_addnoise.pt
```

### 3. Xbox-Controller
The Mujoco Simulator is integerated into ROS2's framework using Xbox's input as commands.
```bash
colcon build --packages-select deploy_sim
source install/setup.bash
ros2 run deploy_sim sim2sim --load_model dt0.001_addnoise.pt
```
Launch the Joy node in another terminal:
```bash
ros2 run joy joy_node
```
