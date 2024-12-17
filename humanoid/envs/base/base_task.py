import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
# 在实例化子类T1DHStandEnv时，子类函数有super继承关系，所以父类函数会初始化，执行init函数，参数为实例化T1DHStandEnv时导入的参数
# self实例指向的子类T1DHStandEnv，所以可以调用子类的函数
class BaseTask():
    # DHT1StandCfg()环境参数
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 负责创建或获取物理仿真对象，它通常会返回一个与仿真环境交互的句柄或对象。
        self.gym = gymapi.acquire_gym()
        # 仿真环境的各种配置参数
        self.sim_params = sim_params
        # SimType.SIM_PHYSX
        self.physics_engine = physics_engine
        # "cuda:0"
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        # 只有当仿真设备是GPU，use_gpu_pipeline=True时，环境设备才是GPU
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1
        # 4096，环境的数量
        self.num_envs = cfg.env.num_envs
        # 47 * 66 = 3102，obs总观察空间的维度 = 单个环境的观察维度 * 帧数
        self.num_obs = cfg.env.num_observations
        # 235 = 47 * 5，critic总观察空间的维度 = 单个环境的观察维度 * 帧数
        self.num_short_obs = int(cfg.env.num_single_obs * cfg.env.short_frame_stack)
        # 219 = 3 * 73
        self.num_privileged_obs = cfg.env.num_privileged_obs
        # 12，动作的维度
        self.num_actions = cfg.env.num_actions
        # 47，单个环境的观察空间维度
        self.num_single_obs = cfg.env.num_single_obs

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        # 创建一个行数为4096，列数为3102的二维全零张量，用于存放66帧观察空间，66 * 47
        # 前者表示环境数量，后者表示每个环境的观察空间的维度，设备类型，数据类型为浮点数
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        # 创建一个行数为4096一维全零张量，用于存放奖励
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # new reward buffers for exp rewrads
        # 创建一个行数为4096一维全零张量，用于存放负奖励
        self.neg_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # 创建一个行数为4096一维全零张量，用于存放正奖励
        self.pos_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # 创建一个行数为4096一维全零张量，用于存放重置环境序号
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # 创建一个行数为4096一维全零张量，用于存放回合长度
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 创建一个行数为4096一维全零张量，用于存放时间超出
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.num_privileged_obs is not None:
            # 创建一个行数为4096，列数为219的二维全零张量，用于存放3帧特权观察空间73
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        # create envs, sim and viewer
        # 创建仿真、地形、环境
        self.create_sim()
        # 解析参数
        self.gym.prepare_sim(self.sim)
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 720
            camera_properties.height = 480
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_properties)
            self.camera_handle = camera_handle
        else:
            # pass
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 720
            camera_properties.height = 480
            camera_handle = self.gym.create_camera_sensor(
                self.envs[0], camera_properties)
            self.camera_handle = camera_handle

    def get_observations(self):
        # 返回一个行数为4096，列数为3102的二维全零张量
        return self.obs_buf

    def get_privileged_observations(self):
        # 返回一个行数为4096，列数为219的二维全零张量
        return self.privileged_obs_buf

    def get_rma_observations(self):
        return self.rma_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
