import os
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.base_task import BaseTask
# from humanoid.utils.terrain import Terrain
# print(f"Actor MLP: {self.actor}")
# print(f"Critic MLP: {self.critic}")
# print(f"long_history CNN: {self.long_history}")
# print(f"state_estimator MLP: {self.state_estimator}")
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from humanoid.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from humanoid.utils.terrain import Terrain

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

# 在实例化子类T1DHStandEnv时，子类函数有super继承关系，所以父类函数会初始化，执行init函数，参数为实例化T1DHStandEnv时导入的参数
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training
        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # DHT1StandCfg()环境参数
        self.cfg = cfg
        # 仿真器参数
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        # 解析参数
        self._parse_cfg(self.cfg)
        # 用于调用父类（BaseTask）的构造函数，并将必要的参数传递给父类，以便父类能正确地初始化自己的状态
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            # 设置相机的位置和方向[22,3,6].[0,3,0]
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # 初始化缓冲区，指向子类self时，调用子类_init_buffers函数
        self._init_buffers()
        # 准备奖励函数
        self._prepare_reward_function()

        self.init_done = True
        self.is_first_add_force = True
        self.is_first_push = True

    # 解析参数，初始化时运行
    def _parse_cfg(self, cfg):
        # 0.01 = 10 * 0.001 策略更新周期 = 决策控制频率（每10个时间步长更新一次策略，输出一次动作） * 时间步长
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        # 观察空间的缩放系数
        self.obs_scales = self.cfg.normalization.obs_scales
        # 包含每个奖励函数的缩放因子的字典
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # 速度的范围字典，线速度、角速度、航向范围
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # 如果地面为plane类型，则不需要课程学习
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        # 每个回合最大持续时间为24s
        self.max_episode_length_s = self.cfg.env.episode_length_s
        # 向上取整（24/0.01） = 2400，每个回合最大长度为2400次策略更新，即2400次动作执行
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # 向上取整（6/0.01）=600，随机化推力的策略间隔，即每6秒施加一次推力，每回合施加4次
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        # 向上取整（6/0.01）=600，随机化外力（推力和扭矩）的时间间隔，即每6秒施加一次推力，每回合施加4次
        self.cfg.domain_rand.ext_force_interval = np.ceil(self.cfg.domain_rand.ext_force_interval_s / self.dt)

    # 初始化缓冲区
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # 获取actor根状态，包括机器人根的位置和姿态
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取自由度（DOF）的状态，表示每个自由度的位置和速度
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取接触力，即物体与环境或其他物体的接触力。
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 获取刚体状态，表示各个刚体（如机器人各个部件）的位置、速度、加速度等。
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # 更新张量，确保每次调用都包含最新数据
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 将根状态和自由度状态张量包装为pytorch张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        # 将dof_state重塑为4096个环境self.num_dof自由度，包含两个数据（位置和速度），提取关节位置
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # 将dof_state重塑为4096个环境self.num_dof自由度，包含两个数据（位置和速度），提取关节速度
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # 从根状态提取四元数，表示机器人根姿态
        self.base_quat = self.root_states[:, 3:7]
        # 根姿态四元数转换为欧拉角
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        # 将接触力张量和刚体状态张量转换为pytorch张量
        # 每个环境中的每个刚体的接触力在XYZ三个轴上的分量，-1 是一个特殊的占位符，在 PyTorch 中使用 -1 表示该维度的大小由 PyTorch 自动推断
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,3)  # shape: num_envs, num_bodies, xyz axis
        # self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, 13, 13)
        # 13是每个刚体的状态信息，包括位置、速度、加速度等。
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        # 提取机器人脚部的四元数信息
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        # 将脚部四元数转换为欧拉角
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        # initialize some data used later on
        # 初始化仿真步骤计数器
        self.common_step_counter = 0
        # 初始化一个空字典extras，用于存储额外信息
        self.extras = {}
        # 获取噪声缩放向量,执行子类的_get_noise_scale_vec函数
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 获取重力向量并转换为张量，get_axis_params 用于确定重力的方向，self.up_axis_idx（2）代表重力方向的轴。然后将重力向量扩展为每个环境的副本
        self.gravity_vec = to_torch(get_axis_params(-1, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 定义机器人的前进方向（在X轴方向），并扩展为每个环境的副本。
        self.forward_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # 初始化一个大小为 (num_envs, num_actions) 的张量 self.torques，用于存储每个环境的力矩信息。
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        # 初始化一个与self.torques相同形状的张量self.last_torques，用于存储上一步的力矩数据。
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        # 初始化一个张量self.actions，存储每个环境的动作信息（通常为控制信号或指令）。
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        # 初始化一个张量 self.last_actions，存储上一步的动作数据。
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        # 初始化一个张量self.last_last_actions，存储前两步的动作数据。
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        # 初始化一个与self.rigid_state形状相同的张量self.last_rigid_state，用于存储上一步的刚体状态。
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        # 初始化一个与 self.contact_forces 形状相同的张量 self.last_contact_forces，用于存储上一步的接触力数据。
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        # 初始化一个与 self.dof_vel 形状相同的张量 self.last_dof_vel，用于存储上一步的关节速度数据。
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # 初始化一个与机器人根部速度（self.root_states[:, 7:13]）形状相同的张量 self.last_root_vel，用于存储上一步的根部速度数据。
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # 创建一个4096 * 4的二维全零矩阵（x轴速度、y轴速度、偏航速度、头偏移）
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        # 初始化一个张量 self.commands_scale，用于存储指令的缩放因子，通常用于将控制命令从某个范围缩放到仿真适用的范围。
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        # 初始化一个张量self.feet_air_time，用于记录每个环境中脚部的空中时间（脚是否离地）。
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        # 初始化一个张量 self.last_contacts，用于记录上一步机器人的脚部接触信息。
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,requires_grad=False)
        # 通过quat_rotate_inverse函数将根部速度转换到机器人局部坐标系下，得到机器人的线速度。
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # 通过quat_rotate_inverse 函数将根部角速度转换到机器人局部坐标系下，得到机器人的角速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 将重力向量转换到机器人的局部坐标系下，self.projected_gravity 用于表示在机器人坐标系下的重力方向。
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 地形是否测量高度，False
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        # 初始化一个张量 self.default_dof_pos，存储每个自由度的默认位置（即机器人的初始关节角度）。
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            # 获取12个关节的名称
            name = self.dof_names[i]

            # 设置默认关节角度
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False

            # 设置PD值
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True

            # 如果没找到PD值，则会初始化为0
            if not found:
                self.p_gains[i] = 0
                self.d_gains[i] = 0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        # 随机推力，形状为self.num_envs * 3
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # 随机扭矩，形状为self.num_envs * 3
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # 额外力
        self.ext_forces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # 额外扭矩
        self.ext_torques = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # 将default_dof_pos在第一个维度上增加一个维度，使其形状变为 1* num_dof
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        # 创建默认关节pd目标值
        self.default_joint_pd_target = self.default_dof_pos.clone()
        # 创建一个队列，用于存储观测历史，队列长度为66，即可以存储66帧的观察空间
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        # 创建一个队列，用于存储评论历史，队列长度为3，即可以存储3帧的critic的观察空间
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        # 将形状为4096 * 47的一个队列添加到self.obs_history，表示一帧的观察空间，一共添加66帧
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))

        for _ in range(self.cfg.env.c_frame_stack):
            # True，即当地形的课程学习打开时
            # 将形状为4096 * 73 + 187的一个队列添加到self.critic_history，表示一帧特权的观察空间，一共3帧
            if self.cfg.terrain.measure_heights:
                self.critic_history.append(torch.zeros(self.num_envs, self.cfg.env.single_num_privileged_obs + self.cfg.terrain.num_height,
                                                       dtype=torch.float, device=self.device))
            else:
            # False，即当课程学习不打开时
            # 将形状为4096 * 73的一个队列添加到self.critic_history，表示一帧特权的观察空间，一共3帧
                self.critic_history.append(torch.zeros(self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

        # True，延迟随机化
        if self.cfg.domain_rand.add_lag:
            # 表示每个环境和每个动作会存储31个滞后时间步的数据。
            self.lag_buffer = torch.zeros(self.num_envs, self.num_actions,self.cfg.domain_rand.lag_timesteps_range[1] + 1, device=self.device)
            # True，表示随机化延迟步数
            if self.cfg.domain_rand.randomize_lag_timesteps:
                # 这一行生成了一个形状为 (self.num_envs,) 的张量 lag_timestep，该张量的每个元素是一个随机整数，范围在lag_timesteps_range[0] 到 lag_timesteps_range[1] 之间
                # 这些随机的滞后时间步数将在不同的环境中引入差异。
                self.lag_timestep = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0],
                                                  self.cfg.domain_rand.lag_timesteps_range[1] + 1, (self.num_envs,),
                                                  device=self.device)
                # 随机化每个时间步
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep = torch.ones(self.num_envs, device=self.device, dtype=int) * \
                                             self.cfg.domain_rand.lag_timesteps_range[1]
            else:
            # 即不启用滞后时间步数的随机化功能，则所有环境的滞后时间步数都设为lag_timesteps_range[1]，即30
                self.lag_timestep = torch.ones(self.num_envs, device=self.device) * \
                                    self.cfg.domain_rand.lag_timesteps_range[1]

        # True，接受信号的延迟随机化(dof_pos和dof_vel),dof_pos和dof_vel延迟一样
        if self.cfg.domain_rand.add_dof_lag:
            # 表示每个环境和每个动作会存储31个滞后时间步的数据。每个动作需要存储两种状态：可能是位置和速度
            self.dof_lag_buffer = torch.zeros(self.num_envs, self.num_actions * 2,
                                              self.cfg.domain_rand.dof_lag_timesteps_range[1] + 1, device=self.device)
            # True，表示随机自由度延迟时间步
            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                # 范围在[0,30]
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                      self.cfg.domain_rand.dof_lag_timesteps_range[1] + 1,
                                                      (self.num_envs,), device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep = torch.ones(self.num_envs, device=self.device, dtype=int) * \
                                                 self.cfg.domain_rand.dof_lag_timesteps_range[1]
            else:
            # 即不启用滞后时间步数的随机化功能，则所有环境的滞后时间步数都设为lag_timesteps_range[1]，即30
                self.dof_lag_timestep = torch.ones(self.num_envs, device=self.device) * \
                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]
        # imu延迟随机化
        if self.cfg.domain_rand.add_imu_lag:
            # 表示每个环境会存储11个滞后时间步的数据，IMU通常会有六个数据通道，分别是加速度和角速度（比如三个轴方向的加速度和三个轴方向的角速度）
            self.imu_lag_buffer = torch.zeros(self.num_envs, 6, self.cfg.domain_rand.imu_lag_timesteps_range[1] + 1,
                                              device=self.device)
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                      self.cfg.domain_rand.imu_lag_timesteps_range[1] + 1,
                                                      (self.num_envs,), device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep = torch.ones(self.num_envs, device=self.device, dtype=int) * \
                                                 self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep = torch.ones(self.num_envs, device=self.device) * \
                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]
        # False，接收信号的延迟随机化(dof_pos和dof_vel),dof_pos和dof_vel延迟不同
        if self.cfg.domain_rand.add_dof_pos_vel_lag:
            self.dof_pos_lag_buffer = torch.zeros(self.num_envs, self.num_actions,
                                                  self.cfg.domain_rand.dof_pos_lag_timesteps_range[1] + 1,
                                                  device=self.device)
            self.dof_vel_lag_buffer = torch.zeros(self.num_envs, self.num_actions,
                                                  self.cfg.domain_rand.dof_vel_lag_timesteps_range[1] + 1,
                                                  device=self.device)
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.dof_pos_lag_timesteps_range[1] + 1,
                                                          (self.num_envs,), device=self.device)
                if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                    self.last_dof_pos_lag_timestep = torch.ones(self.num_envs, device=self.device, dtype=int) * \
                                                     self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]
            else:
                self.dof_pos_lag_timestep = torch.ones(self.num_envs, device=self.device) * \
                                            self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.dof_vel_lag_timesteps_range[1] + 1,
                                                          (self.num_envs,), device=self.device)
                if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                    self.last_dof_vel_lag_timestep = torch.ones(self.num_envs, device=self.device, dtype=int) * \
                                                     self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]
            else:
                self.dof_vel_lag_timestep = torch.ones(self.num_envs, device=self.device) * \
                                            self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]

    # 初始化时运行，准备奖励函数列表，计算奖励函数的总和
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            # 如果奖励尺度为零 (scale == 0)，就从字典中移除这个键值对。
            if scale==0:
                self.reward_scales.pop(key)
            else:
            # 如果奖励尺度非零，乘以dt0.01进行调整，
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        # 奖励函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # 如果奖励名称是 "termination"，则跳过。
            if name=="termination":
                continue
            # 对于其他奖励名称，构造奖励函数的名称（'_reward_' + name），并通过getattr(self, name) 获取相应的奖励函数并添加到 self.reward_functions 列表中。getattr(self, name) 是动态获取对象方法的方式。
            self.reward_names.append(name)
            name = '_reward_' + name
            # 奖励函数的名称
            self.reward_functions.append(getattr(self, name))

        # 用来存储每个奖励名称在每个环境中的累计奖励。每个奖励初始化为零。
        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             # 这是一种 Python 中的字典推导式，用来创建一个新的字典。它会遍历 self.reward_scales.keys() 中的每个奖励名称，并为每个名称生成一个值
                             for name in self.reward_scales.keys()}

    # 返回裁减后的观察值，裁剪后的特许的观察值，奖励值缓冲区（保存当前环境的奖励），重置标志缓冲区（指示是否需要重置某些环境），额外信息（通常用于记录一些附加的调试信息或统计数据）
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 从配置文件中读取，它定义了动作裁剪的范围-100，100
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        # 调用 self.render() 进行环境的渲染，通常用于可视化
        self.render()
        # [0,10]
        for _ in range(self.cfg.control.decimation):
            # 计算扭矩
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 应用扭矩，设置DOF（自由度）的驱动力矩（actuation force），即控制机器人或仿真环境中的关节或物体的力矩。
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 执行仿真，更新仿真状态
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # 刷新仿真中所有自由度（DOF）的状态
            self.gym.refresh_dof_state_tensor(self.sim)
            # True，是否添加接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟一样
            if self.cfg.domain_rand.add_dof_lag:
                # 当前位置
                q = self.dof_pos
                # 当前速度
                dq = self.dof_vel
                self.dof_lag_buffer[:,:,1:] = self.dof_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:,:,0] = torch.cat((q, dq), 1).clone()
            # False，是否添加接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟不同
            if self.cfg.domain_rand.add_dof_pos_vel_lag:
                q = self.dof_pos
                self.dof_pos_lag_buffer[:,:,1:] = self.dof_pos_lag_buffer[:,:,:self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]].clone()
                self.dof_pos_lag_buffer[:,:,0] = q.clone()
                dq = self.dof_vel
                self.dof_vel_lag_buffer[:,:,1:] = self.dof_vel_lag_buffer[:,:,:self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer[:,:,0] = dq.clone()
            # True，是否添加imu的延迟
            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
                self.imu_lag_buffer[:,:,1:] = self.imu_lag_buffer[:,:,:self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.base_euler_xyz ), 1).clone()

        # 确保物理仿真正确地进行，并且相关状态、奖励等信息被更新和记录下来。
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # 100
        clip_obs = self.cfg.normalization.clip_observations
        # [-100,100]
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # 3 * 47 =141
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # 返回裁减后的观察值，裁剪后的特许的观察值，奖励值缓冲区（保存当前环境的奖励），重置标志缓冲区（指示是否需要重置某些环境），额外信息（通常用于记录一些附加的调试信息或统计数据）
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    # 确保物理仿真正确地进行，并且相关状态、奖励等信息被更新和记录下来。
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # 刷新仿真中的不同状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 增加当前回合的长度计数
        self.episode_length_buf += 1
        # 通用计数器
        self.common_step_counter += 1

        # prepare quantities
        # 准备更新的量
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        # 调用自定义的回调函数，执行通用的后处理操作
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # 计算奖励、观察值、是否终止、重置等
        self.check_termination()  # 检查是否有回合结束的条件
        self.compute_reward()# 计算奖励
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()# 获取需要重置的环境ID
        self.reset_idx(env_ids)# 重置相应环境
        # 计算新的观察值,实例化对象为子类，所以self可以当作子类的实例化
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 存储上一步的动作、速度等信息，以便做后续处理
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_torques[:] = self.torques[:]

        # 如果启用了调试可视化并且需要同步显示，调用可视化绘制函数
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # 检查环境是否需要重置
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 检查是否有接触力超出阈值，如果超出，则标记为需要重置的环境True
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1, dim=1)
        # 检查是否超时，如果超出，则标记为需要重置的环境True
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # 将超时和接触力过大都设置为需要重置的环境
        self.reset_buf |= self.time_out_buf

    # 重置环境
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        # True，更新地形课程
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # True and 当前回合步长是最大步长的倍数（一个回合结束），更新命令课程
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        # 重置关节自由度
        self._reset_dofs(env_ids)
        # 重置根状态
        self._reset_root_states(env_ids)
        # 重新采样
        self._resample_commands(env_ids)
    
        # Randomize joint parameters and delay:
        # 随机化机器人的关节参数
        self.randomize_dof_props(env_ids)
        # 刷新刚体的自由度属性
        self._refresh_actor_dof_props(env_ids)
        # 随机化动作延迟属性
        self.randomize_lag_props(env_ids)

        # reset buffers
        # 重置缓冲区
        self.last_last_actions[env_ids] = 0
        self.actions[env_ids] = 0
        self.last_actions[env_ids] = 0
        self.last_torques[env_ids] = 0
        self.last_rigid_state[env_ids] = 0
        self.last_contact_forces[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.last_root_vel[env_ids] = 0
        self.feet_air_time[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # 标记为需要重置的环境
        self.reset_buf[env_ids] = 1

        # fill extras
        # 将与当前回合相关的奖励信息（如累计奖励）存储到self.extras["episode"]字典中
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # plane，记录地形信息
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # 记录命令课程信息，True
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        # None
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # fix reset gravity bug
        # 修复重置引力问题
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 更新物理状态,包括物体的旋转（四元数）、线速度、角速度以及重力等信息
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)
    
    def randomize_lag_props(self,env_ids): 
        if self.cfg.domain_rand.add_lag:   
            self.lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0], 
                                                           self.cfg.domain_rand.lag_timesteps_range[1]+1,(len(env_ids),),device=self.device) 
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]
            else:
                self.lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]
                      
        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep[env_ids] = self.cfg.domain_rand.dof_lag_timesteps_range[1]
            else:
                self.dof_lag_timestep[env_ids] = self.cfg.domain_rand.dof_lag_timesteps_range[1]
 
        if self.cfg.domain_rand.add_imu_lag:                
            self.imu_lag_buffer[env_ids, :, :] = 0.0   
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep[env_ids] = self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep[env_ids] = self.cfg.domain_rand.imu_lag_timesteps_range[1]
    
        if self.cfg.domain_rand.add_dof_pos_vel_lag:
            self.dof_pos_lag_buffer[env_ids, :, :] = 0.0
            self.dof_vel_lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps:
                self.dof_pos_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                    self.last_dof_pos_lag_timestep[env_ids] = self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]
            else:
                self.dof_pos_lag_timestep[env_ids] = self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps:
                self.dof_vel_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                    self.last_dof_vel_lag_timestep[env_ids] = self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]
            else:
                self.dof_vel_lag_timestep[env_ids] = self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]
                      
    # 计算奖励
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # 存储当前回合（episode）中的每个环境的总奖励
        self.rew_buf[:] = 0
        # 奖励函数的数量
        for i in range(len(self.reward_functions)):
            # 奖励函数的名称
            name = self.reward_names[i]
            # 奖励函数本身 * 对应奖励函数的缩放因子
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # 奖励值累加，表示当前回合所有环境的总奖励。
            self.rew_buf += rew
            # 将每个奖励项rew累加到self.episode_sums[name]中，记录某个特定奖励项的累计和。
            self.episode_sums[name] += rew
        # Ture，对奖励值进行限制（只允许正奖励），负奖励值为0
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0)
        # add termination reward after clipping

        # None，缩放字典中没有termination属性
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # 设置相机的位置
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    # 刚体属性随机化
    def randomize_rigid_body_props(self, env_ids):
        ''' Randomise some of the rigid body properties of the actor in the given environments, i.e.
            sample the mass, centre of mass position, friction and restitution.'''
        # From Walk These Ways:
        # 基座质量随机化
        if self.cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = self.cfg.domain_rand.added_mass_range
            self.payload_masses[env_ids] = torch_rand_float(min_payload, max_payload, (len(env_ids), 1), device=self.device)
        # 连杆质量随机化
        if self.cfg.domain_rand.randomize_link_mass:
            min_link_mass, max_link_mass = self.cfg.domain_rand.added_link_mass_range
            self.link_masses[env_ids] = torch_rand_float(min_link_mass, max_link_mass, (len(env_ids), self.num_bodies-1), device=self.device)
        # 质心随机化
        if self.cfg.domain_rand.randomize_com:
            comx_displacement, comy_displacement, comz_displacement = self.cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.cat((torch_rand_float(comx_displacement[0], comx_displacement[1], (len(env_ids), 1), device=self.device),
                                                            torch_rand_float(comy_displacement[0], comy_displacement[1], (len(env_ids), 1), device=self.device),
                                                            torch_rand_float(comz_displacement[0], comz_displacement[1], (len(env_ids), 1), device=self.device)),
                                                            dim=-1)
        # 连杆质心随机化
        if self.cfg.domain_rand.randomize_link_com:
            comx_displacement, comy_displacement, comz_displacement = self.cfg.domain_rand.link_com_displacement_range
            self.link_com_displacements[env_ids, :, :] = torch.cat((torch_rand_float(comx_displacement[0], comx_displacement[1], (len(env_ids), self.num_bodies-1, 1), device=self.device),
                                                                    torch_rand_float(comy_displacement[0], comy_displacement[1], (len(env_ids), self.num_bodies-1, 1), device=self.device),
                                                                    torch_rand_float(comz_displacement[0], comz_displacement[1], (len(env_ids), self.num_bodies-1, 1), device=self.device)),
                                                                    dim=-1)
        # 基座惯性矩阵随机化
        if self.cfg.domain_rand.randomize_base_inertia:
            inertia_x, inertia_y, inertia_z = self.cfg.domain_rand.base_inertial_range
            self.base_inertia_x[env_ids, :, :] = torch_rand_float(inertia_x[0], inertia_x[1], (len(env_ids), 1), device=self.device)
            self.base_inertia_y[env_ids, :, :] = torch_rand_float(inertia_y[0], inertia_y[1], (len(env_ids), 1), device=self.device)
            self.base_inertia_z[env_ids, :, :] = torch_rand_float(inertia_z[0], inertia_z[1], (len(env_ids), 1), device=self.device)
        # 连杆惯性矩阵随机化
        if self.cfg.domain_rand.randomize_link_inertia:
            inertia_x, inertia_y, inertia_z = self.cfg.domain_rand.link_inertial_range
            self.link_inertia_x[env_ids, :, :] = torch_rand_float(inertia_x[0], inertia_x[1], (len(env_ids), self.num_bodies-1), device=self.device)
            self.link_inertia_y[env_ids, :, :] = torch_rand_float(inertia_y[0], inertia_y[1], (len(env_ids), self.num_bodies-1), device=self.device)
            self.link_inertia_z[env_ids, :, :] = torch_rand_float(inertia_z[0], inertia_z[1], (len(env_ids), self.num_bodies-1), device=self.device)

    # 自由度属性随机化
    def randomize_dof_props(self, env_ids):
        # Randomise the motor strength:
        # 扭矩随机化
        if self.cfg.domain_rand.randomize_torque:
            motor_strength_ranges = self.cfg.domain_rand.torque_multiplier_range
            self.torque_multi[env_ids] = torch_rand_float(motor_strength_ranges[0], motor_strength_ranges[1], (len(env_ids),self.num_actions), device=self.device)
        # 电机补偿随机化
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch_rand_float(min_offset, max_offset, (len(env_ids),self.num_actions), device=self.device)
        # pd增益随机化
        if self.cfg.domain_rand.randomize_gains:
            p_gains_range = self.cfg.domain_rand.stiffness_multiplier_range
            d_gains_range = self.cfg.domain_rand.damping_multiplier_range
            self.randomized_p_gains[env_ids] = torch_rand_float(p_gains_range[0], p_gains_range[1], (len(env_ids),self.num_actions), device=self.device) * self.p_gains
            self.randomized_d_gains[env_ids] =  torch_rand_float(d_gains_range[0], d_gains_range[1], (len(env_ids),self.num_actions), device=self.device) * self.d_gains    
        # 库伦摩擦力随机化
        if self.cfg.domain_rand.randomize_coulomb_friction:
            joint_coulomb_range = self.cfg.domain_rand.joint_coulomb_range
            joint_viscous_range = self.cfg.domain_rand.joint_viscous_range
            self.randomized_joint_coulomb[env_ids] = torch_rand_float(joint_coulomb_range[0], joint_coulomb_range[1], (len(env_ids),self.num_actions), device=self.device)
            self.randomized_joint_viscous[env_ids] =  torch_rand_float(joint_viscous_range[0], joint_viscous_range[1], (len(env_ids),self.num_actions), device=self.device)  
        # 关节摩擦力随机化
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_friction_range'
                    friction_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_friction_coeffs[env_ids, i] = torch_rand_float(friction_range[0], friction_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:                      
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)
        # 关节阻尼随机化
        if self.cfg.domain_rand.randomize_joint_damping:
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_damping_range'
                    damping_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_damping_coeffs[env_ids, i] = torch_rand_float(damping_range[0], damping_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_damping_range = self.cfg.domain_rand.joint_damping_range
                self.joint_damping_coeffs[env_ids] = torch_rand_float(joint_damping_range[0], joint_damping_range[1], (len(env_ids), 1), device=self.device)
        # 关节电枢随机化
        if self.cfg.domain_rand.randomize_joint_armature:
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_armature_range'
                    armature_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_armatures[env_ids, i] = torch_rand_float(armature_range[0], armature_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                self.joint_armatures[env_ids] = torch_rand_float(joint_armature_range[0], joint_armature_range[1], (len(env_ids), 1), device=self.device)

    # 实现了对每个环境的摩擦系数（friction）和反弹系数（restitution）的随机化功能
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment
        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id
        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # 摩擦力随机化
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                # 摩擦系数随机化范围
                friction_range = self.cfg.domain_rand.friction_range
                # 反弹系数随机化范围
                restitution_range = self.cfg.domain_rand.restitution_range
                # 创建256个桶，每个桶对应一个摩擦系数和反弹系数的随机值范围。这是为了将摩擦和反弹系数预先分配到桶中，随机选择桶来为每个环境提供摩擦和反弹系数。
                num_buckets = 256
                # 为每个环境生成一个随机的桶ID，表示从这些桶中选择摩擦系数和反弹系数。
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                # 为每个桶生成一个随机的摩擦系数，在指定的范围内。
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                # 为每个桶生成一个随机的反弹系数。
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                # 根据为每个环境生成的桶ID，选择对应的摩擦和反弹系数。
                self.friction_coeffs = friction_buckets[bucket_ids]
                self.restitution_coeffs = restitution_buckets[bucket_ids]
            # 更新每个刚体的摩擦和反弹系数
            for s in range(len(props)):
                # 将当前环境的摩擦系数（根据env_id获取）赋值给每个刚体形状的摩擦属性。
                props[s].friction = self.friction_coeffs[env_id]
                # 将当前环境的反弹系数赋值给每个刚体形状的反弹属性。
                props[s].restitution = self.restitution_coeffs[env_id]
            # 将当前环境的摩擦系数记录到 self.env_frictions 中，以便在其他地方使用或调试
            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        # 返回修改后的props列表
        return props

    # 在环境创建过程中处理每个环境中的自由度（DOF）属性
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF
        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id
        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                # 表示每个自由度的最小位置限制
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                # 表示每个自由度的最大位置限制
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                # 表示每个自由度的速度限制
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                # 表示每个自由度的扭矩（力矩）限制
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self.init_body_mass[env_id] = props[0].mass
            props[0].mass += self.payload_masses[env_id]
        self.body_mass[env_id] = props[0].mass
            
        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(1, len(props)):
                props[i].mass *= self.link_masses[env_id, i-1]
        for i in range(1, len(props)):    
            self.total_mass[env_id] += props[i].mass
        
        if self.cfg.domain_rand.randomize_com:
             props[0].com += gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                    self.com_displacements[env_id, 2])
             
        if self.cfg.domain_rand.randomize_link_com:
            for i in range(1, len(props)):
                props[i].com += gymapi.Vec3(self.link_com_displacements[env_id, i-1, 0], self.link_com_displacements[env_id, i-1, 1],
                                           self.link_com_displacements[env_id, i-1, 2])     
                
        if self.cfg.domain_rand.randomize_base_inertia:
            props[0].inertia.x.x *= self.base_inertia_x[env_id]
            props[0].inertia.y.y *= self.base_inertia_y[env_id]
            props[0].inertia.z.z *= self.base_inertia_z[env_id]
                
        if self.cfg.domain_rand.randomize_link_inertia:
            for i in range(1, len(props)):
                props[i].inertia.x.x *= self.link_inertia_x[env_id, i-1]
                props[i].inertia.y.y *= self.link_inertia_y[env_id, i-1]
                props[i].inertia.z.z *= self.link_inertia_z[env_id, i-1]
                
        return props
    
    def randomize_rigid_props(self,env_ids):
        if self.cfg.domain_rand.randomize_friction:
            min_friction, max_friction = self.cfg.domain_rand.friction_range
            self.friction[env_ids, :] = torch_rand_float(min_friction, max_friction, (len(env_ids), 1), device=self.device)
            
    def _refresh_actor_rigid_shape_props(self, env_ids):
        ''' Refresh the rigid shape properties of the actor in the given environments, i.e.
            set the friction and restitution coefficients to the desired values.
        '''
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _refresh_actor_rigid_body_props(self, env_ids):
        ''' Refresh the rigid body properties of the actor in the given environments, i.e.
            set the payload_mass to the desired values.
        '''
        for env_id in env_ids:
            rigid_body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], 0)

            rigid_body_props[0].mass = self.payload_masses[env_id] + self.init_body_mass[env_id]

            self.gym.set_actor_rigid_body_properties(self.envs[env_id], 0, rigid_body_props)
    
    def _refresh_actor_dof_props(self, env_ids):

        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                if self.cfg.domain_rand.randomize_joint_friction:
                    if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, i]
                    else:    
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_joint_damping:
                    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, i]
                    else:
                        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, 0]
                        # print(dof_props["damping"])
                        
                if self.cfg.domain_rand.randomize_joint_armature:
                    if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                        dof_props["armature"][i] = self.joint_armatures[env_id, i]
                    else:
                        dof_props["armature"][i] = self.joint_armatures[env_id, 0]
                # print(dof_props["friction"][i], dof_props["damping"][i], dof_props["armature"][i])
            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)

    # 通常包含一些在每一步物理仿真后需要执行的通用任务。比如状态更新、传感器数据采集等。
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 判断当前的步数是否符合重新采样控制命令的条件
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # 用于为多个环境随机生成控制命令的函数
        self._resample_commands(env_ids)
        # False,是否根据头部偏差计算角速度
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        # False，调用 _get_heights() 方法来获取当前环境中地形的高度信息。
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # False，是否配置启用了随机推送机器人
        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter/self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()
                self.is_first_push = True
        # False，是否添加外力
        if self.cfg.domain_rand.add_ext_force:
            i = int(self.common_step_counter/self.cfg.domain_rand.add_update_step)
            if i >= len(self.cfg.domain_rand.add_duration):
                i = len(self.cfg.domain_rand.add_duration) - 1
            duration = self.cfg.domain_rand.add_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.ext_force_interval <= duration:
                self._add_ext_force()
            else:
                self.ext_forces.zero_()
                self.ext_torques.zero_()
                self.is_first_add_force = True
    # 添加外力
    def _add_ext_force(self):
        apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        apply_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        if self.is_first_add_force:
            force_x = torch_rand_float(-self.cfg.domain_rand.ext_force_max_x/2, self.cfg.domain_rand.ext_force_max_x,(self.num_envs,1),device=self.device)
            force_y = torch_rand_float(-self.cfg.domain_rand.ext_force_max_y, self.cfg.domain_rand.ext_force_max_y,(self.num_envs,1),device=self.device)
            force_z = torch_rand_float(-self.cfg.domain_rand.ext_force_max_z, self.cfg.domain_rand.ext_force_max_z,(self.num_envs,1),device=self.device)
            self.ext_forces = torch.cat((force_x, force_y, force_z), 1)
            self.ext_torques = torch_rand_float(-self.cfg.domain_rand.ext_torque_max, self.cfg.domain_rand.ext_torque_max,(self.num_envs,3),device=self.device)
        if self.is_first_add_force == False:
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            apply_forces[:, 0, :] = self.ext_forces * stand_command.unsqueeze(-1)
            apply_torques[:, 0, :] = self.ext_torques * stand_command.unsqueeze(-1)
        self.is_first_add_force = False
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(apply_forces), gymtorch.unwrap_tensor(apply_torques), gymapi.ENV_SPACE)

    # 用于为多个环境随机生成控制命令的函数
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # 随机选择线速度命令（x和y轴）
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # False，是否根据头偏差计算角速度
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            # 随机选择角速度
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 对于小于 0.2 的速度命令，将其置为零，防止过小的命令影响控制效果。
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        # self.commands[env_ids, 2] *= (torch.norm(self.commands[env_ids, 2], dim=1) > 0.2).unsqueeze(1)

    # 计算扭矩
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        # 动作缩减一半
        actions_scaled = actions * self.cfg.control.action_scale
        # True，在动作中引入延迟
        if self.cfg.domain_rand.add_lag:
            self.lag_buffer[:,:,1:] = self.lag_buffer[:,:,:self.cfg.domain_rand.lag_timesteps_range[1]].clone()
            # 存储每个环境在不同时间步的动作历史
            self.lag_buffer[:,:,0] = actions_scaled.clone()
            # False，是否在每个环境步骤中随机化延迟的时间步数
            if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                self.lag_timestep = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.lag_timestep > self.last_lag_timestep + 1
                self.lag_timestep[cond] = self.last_lag_timestep[cond] + 1
                self.last_lag_timestep = self.lag_timestep.clone()
            # 根据随机化的延迟时间步数lag_timestep从lag_buffer中获取实际使用的动作。
            self.lagged_actions_scaled = self.lag_buffer[torch.arange(self.num_envs),:,self.lag_timestep.int()]
        else:
            self.lagged_actions_scaled = actions_scaled
        # False，是否随机化增益
        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains
        # Ture，随机化库伦摩擦力
        if self.cfg.domain_rand.randomize_coulomb_friction:
            # kp * (延迟步数实际使用的动作+默认关节角度+补偿-当前关节角度) - kd当前速度
            torques = p_gains * (self.lagged_actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets) -\
            d_gains * self.dof_vel -\
            self.randomized_joint_viscous * self.dof_vel - self.randomized_joint_coulomb * torch.sign(self.dof_vel)
        else: 
            torques = p_gains * (self.lagged_actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets) - d_gains * self.dof_vel
        
        # np.set_printoptions(formatter={'float': '{:0.4f}'.format})
        # print("\ntau", torques[0].cpu().numpy())
        # print("q", self.dof_pos[0].cpu().numpy())

        # False，是否随机化扭矩
        if self.cfg.domain_rand.randomize_torque:
            motor_strength_ranges = self.cfg.domain_rand.torque_multiplier_range
            self.torque_multi = torch_rand_float(motor_strength_ranges[0], motor_strength_ranges[1], (self.num_envs,self.num_actions), device=self.device)
            torques *= self.torque_multi
        # 根据输入的动作计算出扭矩，并返回给物理引擎，限制扭矩的范围，使其不超过self.torque_limits设置的最大和最小值
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.terrain.curriculum:
                platform = self.cfg.terrain.platform
                self.root_states[env_ids, :2] += torch_rand_float(-platform/3, platform/3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            else:
                terrain_length = self.cfg.terrain.terrain_length
                self.root_states[env_ids, :2] += torch_rand_float(-terrain_length/2, terrain_length/2, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.01, 0.01, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    # 推动机器人
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.is_first_push:
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            max_push_angular = self.cfg.domain_rand.max_push_ang_vel
            self.rand_push_force[:, :2] = torch_rand_float(
                -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
            self.rand_push_torque = torch_rand_float(
                -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)  #angular vel xyz
        self.is_first_push = False
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]
        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.25, -self.cfg.commands.max_curriculum/2, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    # 创建plane地形
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        # 创建地面平面的参数对象
        plane_params = gymapi.PlaneParams()
        # 设置地面平面的法向量（平面朝上的方向）
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # 设置地面平面的静态摩擦力（即物体静止时的摩擦力）
        plane_params.static_friction = self.cfg.terrain.static_friction
        # 设置地面平面的动态摩擦力（即物体滑动时的摩擦力）
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        # 设置地面平面的恢复系数（即碰撞后物体反弹的弹性，0 表示完全无弹性，1 表示完全弹性）
        plane_params.restitution = self.cfg.terrain.restitution
        # 将地面平面添加到仿真环境中
        self.gym.add_ground(self.sim, plane_params)

    # 创建heightfield地形
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        # 创建一个高程场参数对象
        hf_params = gymapi.HeightFieldParams()
        # 设置高程场的水平尺度（列和行的比例）
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        # 设置高程场的垂直尺度
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        # 设置高程场的行数和列数
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        # 设置高程场的变换（位置、坐标偏移）
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        # 设置高程场的摩擦力和恢复系数
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        # 将高程场添加到仿真环境中
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        # 将高程样本数据转换为torch张量并保存
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # 创建trimesh地形
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        # 创建一个三角网格地形的参数对象
        tm_params = gymapi.TriangleMeshParams()
        # 设置三角网格的顶点数量
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        # 设置三角网格的三角形数量
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        # 设置三角网格的变换（位置、坐标偏移）
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        # 设置三角网格的摩擦力和恢复系数
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        # 将三角网格地形添加到仿真环境中
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        # 将高程样本数据转换为PyTorch张量并保存
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        # 创建一个仿真环境实例，在该实例中将使用指定的设备、物理引擎和仿真参数进行计算和模拟
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    # 创建机器人仿真环境，配置机器人模型，并将其加载到仿真环境中
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # 获取机器人URDF模型
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # 获取文件的目录路径
        asset_root = os.path.dirname(asset_path)
        # 获取文件的名字
        asset_file = os.path.basename(asset_path)

        # 创建一个属性对象
        asset_options = gymapi.AssetOptions()
        # 3
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # True
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # True
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # True
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # False
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # 0.001
        asset_options.density = self.cfg.asset.density
        # 0
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 0
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 1000
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 1000
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 0
        asset_options.armature = self.cfg.asset.armature
        # 0.01
        asset_options.thickness = self.cfg.asset.thickness
        # False
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        # 加载机器人模型
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # 获取机器人模型属性
        # 获取机器人自由度数量
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        # 获取机器人刚体数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        # 获取机器人自由度属性
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # 获取机器人刚体数量属性
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        # 获取机器人刚体的名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # 获取机器人自由度的名称
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # 获取刚体名称的长度，13
        self.num_bodies = len(body_names)
        # 获取自由度名称的长度，12
        self.num_dofs = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        penalized_contact_names = []
        # base_link
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        # base_link
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 初始化机器人状态，位置 + 姿态 + 线速度 + 角速度
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # 机器人的初始变换，包含了位置和旋转信息
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # 创建一个全零长度为12的一维张量
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 创建一个全零长度为12的一维矩阵
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # 为仿真环境中的各种物理属性（如机器人各个部分的质量、惯性、摩擦力等）生成随机化的值
        self.init_randomize_props()

        # 设置仿真环境中机器人的原点位置
        self._get_env_origins()

        # 设置了每个仿真环境的边界（这里设定为零表示没有限制）
        env_lower = gymapi.Vec3(0, 0, 0)
        env_upper = gymapi.Vec3(0, 0, 0)
        # 用于存储机器人的句柄
        self.actor_handles = []
        # 用于存储环境句柄
        self.envs = []
        # 创建摩擦力的一维矩阵，大小为4096 * 1
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        # 创建质量的一维矩阵，大小为4096 * 1
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        # 创建初始质量的一维矩阵，大小为4096 * 1
        self.init_body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        # 创建总质量的一维矩阵，大小为4096 * 1
        self.total_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)

        # 刚体属性随机化
        self.randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))
        # 自由度属性随机化
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))

        for i in range(self.num_envs):
            # create env instance
            # 创建每个仿真环境，环境的大小由env_lower和env_upper定义。
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 设置环境初始位置
            pos = self.env_origins[i].clone()
            # 机器人的位置pos被稍微随机化（torch_rand_float(-1., 1., (2,1))），使得机器人在每个环境中的初始位置稍有不同
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            # 实现了对每个环境的摩擦系数（friction）和反弹系数（restitution）的随机化功能
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # 将修改的刚体形状属性值添加到机器人模型上
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 创建机器人actor，并将其添加到指定的环境中
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # 在环境创建过程中处理每个环境中的自由度（DOF）属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            # 将修改的自由度属性值添加到机器人模型上
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            # 获取并处理机器人的刚体属性，并将其应用于仿真环境中的机器人actor
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # 表示仿真环境的引用
            self.envs.append(env_handle)
            # 代表仿真中的物理对象（如机器人、角色）的引用
            self.actor_handles.append(actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # 为仿真环境中的各种物理属性（如机器人各个部分的质量、惯性、摩擦力等）生成随机化的值
    def init_randomize_props(self):
        # 基座质量随机化
        if self.cfg.domain_rand.randomize_base_mass:
            self.payload_masses = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        # 质心随机化
        if self.cfg.domain_rand.randomize_com:
            self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        # 基座惯性矩镇随机化
        if self.cfg.domain_rand.randomize_base_inertia:
            self.base_inertia_x = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.base_inertia_y = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.base_inertia_z = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # 连杆质量随机化
        if self.cfg.domain_rand.randomize_link_mass:
            self.link_masses = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device,requires_grad=False)
        # 连杆质心随机化
        if self.cfg.domain_rand.randomize_link_com:
            self.link_com_displacements = torch.zeros(self.num_envs, self.num_bodies-1, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 连轧惯性矩阵随机化
        if self.cfg.domain_rand.randomize_link_inertia:
            self.link_inertia_x = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
            self.link_inertia_y = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
            self.link_inertia_z = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
        # 摩擦力随机化
        if self.cfg.domain_rand.randomize_friction:
            self.friction = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)     
        # 每个关节的关节摩擦力随机化
        if self.cfg.domain_rand.randomize_joint_friction_each_joint:
            self.joint_friction_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
        else:
            self.joint_friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        # 每个关节的关节阻尼随机化
        if self.cfg.domain_rand.randomize_joint_damping_each_joint:
            self.joint_damping_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
        else:
            self.joint_damping_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        # 每个关节的关节电枢随机化
        if self.cfg.domain_rand.randomize_joint_armature_each_joint:
            self.joint_armatures = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)  
        else:
            self.joint_armatures = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        # 扭矩随机化
        if self.cfg.domain_rand.randomize_torque:
            self.torque_multi = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        # 创建一个全零，大小为4096 * 12 的二维张量
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False) 
        # pd增益随机化
        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.p_gains
            self.randomized_d_gains = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.d_gains
        # 库伦摩擦力随机化
        if self.cfg.domain_rand.randomize_coulomb_friction:
            self.randomized_joint_coulomb = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.p_gains
            self.randomized_joint_viscous = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.d_gains

    # 设置仿真环境中机器人的起始位置
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # 创建一个形状为4096 * 3的张量self.env_origins，用于存储每个环境中机器人的原点位置，初始化为零
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            # 计算机器人在地形中的起始位置
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            # 获取地形原点并更新机器人位置
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            # 据机器人环境的数量 (num_envs)，计算网格的列数。列数是num_envs的平方根
            num_cols = np.floor(np.sqrt(self.num_envs))
            # 据机器人环境的数量 (num_envs)，计算网格的行数。行数是num_envs的平方根
            num_rows = np.ceil(self.num_envs / num_cols)
            # 创建一个网格，生成机器人放置的行列坐标。
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            # 从配置文件中获取网格中机器人的间距
            spacing = self.cfg.env.env_spacing
            # 将网格的x坐标分配给机器人原点
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            # 将网格的y坐标分配给机器人原点
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            # 将机器人的z坐标设置为零，意味着机器人放置在平面上
            self.env_origins[:, 2] = 0

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
