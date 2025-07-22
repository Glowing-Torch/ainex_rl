# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
from rclpy.node import Node
import rclpy
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import XBotLCfg
from humanoid.deploy_sim.deploy_sim.xbox_command import XboxController
import threading
import torch
import time

class Sim2simCfg(XBotLCfg):

    class sim_config:
        mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/ainex_description/mjcf/ainex.xml'
        sim_duration = 60.0
        dt = 0.001
        decimation = 10

    class robot_config:
        kps = np.array([4, 6, 4, 4, 2, 2,
                        4, 6, 4, 4, 2, 2], dtype=np.double)
        kds = np.array([0.1,0.1,0.1,0.1, 0.3,0.1, 
                        0.1,0.1,0.1,0.1, 0.3,0.1], dtype=np.double)

        tau_limit = 5.1 * np.ones(12, dtype=np.double)
        
class MujocoSimulator(Node):
    def __init__(self,model_path):
        super().__init__('mujoco_simulator')
        self.cmd_sub = XboxController(self)
        self.cfg=Sim2simCfg()
        self.policy = torch.jit.load(model_path)
        self.policy.eval()
        self.running = True
        self.init_mujoco()
        self.sim_thread = threading.Thread(target=self.step_simulation)
        self.sim_thread.start()
    
    def init_mujoco(self):        
        self.m = mujoco.MjModel.from_xml_path(str(self.cfg.sim_config.mujoco_model_path))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.cfg.sim_config.dt
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        print("Number of qpos:", self.m.nq)
        print("Joint order:")
        for i in range(self.m.njnt):
            print(f"{i}: {self.m.joint(i).name}")
            
    def step_simulation(self):
        target_q = np.zeros((self.cfg.env.num_actions), dtype=np.double)
        action = np.zeros((self.cfg.env.num_actions), dtype=np.double)
        default_joint_pos = np.array([0,0,-0.293,0.376,0.0836,0,
                                     0,0, 0.293, -0.376,-0.0836,0])
        # default_joint_pos = np.zeros((cfg.env.num_actions), dtype=np.double)
        hist_obs = deque()
        for _ in range(self.cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))
        count_lowlevel = 0
        cmd= np.zeros((3), dtype=np.float32)
        while self.viewer.is_running() and self.running:
            step_start=time.time()
            # Obtain an observation
            q, dq, quat, omega = get_obs(self.d)
            # 1000hz -> 100hz
            if self.cmd_sub.is_pressed():
                linear_x, linear_y =self.cmd_sub.get_left_stick()
                angular_z = self.cmd_sub.get_right_stick()
                cmd=np.array([linear_x, linear_y, angular_z], dtype=np.float32)
            if count_lowlevel % self.cfg.sim_config.decimation == 0:
                obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * self.cfg.sim_config.dt / 0.5)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * self.cfg.sim_config.dt / 0.5)
                obs[0, 2] = cmd[0] * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = cmd[1] * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = cmd[2] * self.cfg.normalization.obs_scales.ang_vel
                obs[0, 5:17] = (q - default_joint_pos) * self.cfg.normalization.obs_scales.dof_pos
                obs[0, 17:29] = dq * self.cfg.normalization.obs_scales.dof_vel
                obs[0, 29:41] = action
                obs[0, 41:44] = omega
                obs[0, 44:47] = eu_ang
                obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
                hist_obs.append(obs)
                hist_obs.popleft()
                policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
                for i in range(self.cfg.env.frame_stack):
                    policy_input[0, i * self.cfg.env.num_single_obs : (i + 1) * self.cfg.env.num_single_obs] = hist_obs[i][0, :]
                with torch.no_grad():
                    action[:] = self.policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
                target_q = action * self.cfg.control.action_scale + default_joint_pos

            target_dq = np.zeros((self.cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, self.cfg.robot_config.kps,
                            target_dq, dq, self.cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)  # Clamp torques
            self.d.ctrl = tau

            mujoco.mj_step(self.m, self.d)
            self.viewer.sync()
            count_lowlevel += 1
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.viewer.close()
    
    def stop_simulation(self):
        self.running = False
        self.sim_thread.join()
        
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    
    q = data.qpos.astype(np.double)[7:19]
    dq = data.qvel.astype(np.double)[6:18]
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    return (q, dq, quat, omega)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # print("target_q", target_q)
    # print("q", q)
    # print("kp", kp)
    return (target_q - q) * kp + (target_dq - dq) * kd



def main():
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/Ainex/exported/policies/{args.load_model}'
    rclpy.init()
    node = MujocoSimulator(args.load_model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_simulation()
        node.destroy_node()
        rclpy.shutdown()
