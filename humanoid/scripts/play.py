import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

import pygame
from threading import Thread

import time

# 机器人的指令
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
# x_scale, y_scale, yaw_scale = 2.5, 2.0, 0.0

joystick_use = True
joystick_opened = False

# True
if joystick_use:
    # 初始化pygame库，用于处理游戏控制器的输入
    pygame.init()
    try:
        # 获取第一个手柄（索引号为0）
        joystick = pygame.joystick.Joystick(0)
        # 初始化手柄
        joystick.init()
        # 标记手柄已打开成功
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
    # 用于控制线程退出的标志
    exit_flag = False
    # 处理手柄输入的线程
    def handle_joystick_input():
        # 全局变量
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        while not exit_flag:
            # 获取手柄输入信号
            pygame.event.get()
            # 获取手柄的各个轴的值并更新机器人的速度命令
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1
            # 打印出控制命令
            print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
            # 等待一小段时间，防止程序过度占用CPU，可以根据实际情况调整
            pygame.time.delay(100)
    if joystick_opened and joystick_use:
        # 创建一个新线程，目标是执行'handle_joystick_input'函数
        joystick_thread = Thread(target=handle_joystick_input)
        # 启动线程
        joystick_thread.start()

def play(args):
    # 创建环境配置DHT1StandCfg()对象和训练配置DHT1StandCfgPPO()训练参数对象
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # 设置机器人初始位置
    env_cfg.init_state.pos = [0.0, 0.0, 1.1]
    # 机器人的数量
    env_cfg.env.num_envs = 9
    # env_cfg.sim.max_gpu_contact_pairs = 2**10

    # 地形类型设置为平面地形，trimesh为三角网格模型
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    # 地形行数和数量
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = True     
    env_cfg.terrain.max_init_terrain_level = 5

    # 表示仿真运行的时间1000s，超过时间就会重置
    env_cfg.env.episode_length_s = 1000
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False 
    # env_cfg.domain_rand.push_robots = False 
    # env_cfg.domain_rand.continuous_push = False 
    # env_cfg.domain_rand.randomize_base_mass = False 
    # env_cfg.domain_rand.randomize_com = False 
    # env_cfg.domain_rand.randomize_gains = False 
    # env_cfg.domain_rand.randomize_torque = False 
    # env_cfg.domain_rand.randomize_link_mass = False 
    # env_cfg.domain_rand.randomize_motor_offset = False 
    # env_cfg.domain_rand.randomize_joint_friction = False
    # env_cfg.domain_rand.randomize_joint_damping = False
    # env_cfg.domain_rand.randomize_joint_armature = False
    # env_cfg.domain_rand.randomize_lag_timesteps = False
    # env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.domain_rand.add_lag =True
    env_cfg.domain_rand.randomize_lag_timesteps = True
    env_cfg.domain_rand.add_dof_lag = True
    env_cfg.domain_rand.randomize_dof_lag_timesteps = True
    env_cfg.domain_rand.add_imu_lag = True
    env_cfg.domain_rand.randomize_imu_lag_timesteps = True
    env_cfg.noise.curriculum = False
    # env_cfg.noise.noise_level = 1.0
    # env_cfg.sim.dt = 0.0005
    # env_cfg.sim.sim_duration = 60
    # env_cfg.control.decimation = 40
    env_cfg.commands.heading_command = False

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    # env是创建的T1DHStandEnv环境对象，env_cfg是返回环境参数的DHT1StandCfg对象
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 设置相机位置和方向
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy，加载训练好的模型
    train_cfg.runner.resume = True
    # 创建DHOnPolicyRunner对象、DHT1StandCfgPPO()对象、logs存储路径
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 创建实例接口，根据当前观察值，返回动作的平均数
    policy = ppo_runner.get_inference_policy(device=env.device)

    print(datetime.now().strftime('%Y-%m-%d'+' '+'%H-%M-%S'))
    
    # export policy as a jit module (used to run it from C++)
    # 将策略导出为jit模块（用于从C++运行它）
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    #  0.001 * 10 = 0.01，策略更新周期/控制周期
    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    # 指定机器人的索引号
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    # 画图的时间2000 * 0.01 = 20 s
    stop_state_log = 2000 # number of steps before plotting states

    # 在一个仿真环境中设置相机并准备录制视频
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)

        # 相机位置姿态
        camera_offset = gymapi.Vec3(6, 0, 2)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.2, 0, 1),np.deg2rad(180))

        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        # 录制视频名称
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')

        if not os.path.exists(video_dir):
            os.makedirs(video_dir,exist_ok=True)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir,exist_ok=True)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    # 获取观察空间，4096 * 3102
    obs = env.get_observations()
    # 这行代码的目的是修改NumPy数组的打印方式，使得所有浮点数显示时都保留4位小数。
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    # 运行指定迭代次数
    for i in range(10*stop_state_log):

        # 传入动作空间的张量，返回动作的平均值
        actions = policy(obs.detach()) # * 0

        if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0
            env.commands[:, 2] = 0
            env.commands[:, 3] = 0
        else:
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = 0

        # 传入actions的张量，返回actor的观察空间、critic的观察空间、奖励值、终止信号、其他信息
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        # 更新仿真状态，获取当前相机的图像，处理该图像，并将其添加到视频中。
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        if i < stop_state_log:
            dict = {
                    'base_height' : env.root_states[robot_index, 2].item(),
                    'foot_z_l' : env.rigid_state[robot_index,4,2].item(),
                    'foot_z_r' : env.rigid_state[robot_index,9,2].item(),
                    'foot_forcez_l' : env.contact_forces[robot_index,env.feet_indices[0],2].item(),
                    'foot_forcez_r' : env.contact_forces[robot_index,env.feet_indices[1],2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'command_x': x_vel_cmd,
                    'base_vel_y':  env.base_lin_vel[robot_index, 1].item(),
                    'command_y': y_vel_cmd,
                    'base_vel_z':  env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw':  env.base_ang_vel[robot_index, 2].item(),
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, 0].item(),
                    'dof_vel': env.dof_vel[robot_index, 0].item(),
                    'dof_torque': env.torques[robot_index, 0].item(),
                }

            # 添加 dof_pos_target 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos_target[{i}]'] = actions[robot_index, i].item() * env.cfg.control.action_scale,

            # 添加 dof_pos 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos[{i}]'] = env.dof_pos[robot_index, i].item(),

            # 添加 dof_torque 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_torque[{i}]'] = env.torques[robot_index, i].item(),

            # 添加 dof_vel 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_vel[{i}]'] = env.dof_vel[robot_index, i].item(),

            logger.log_states(dict=dict)

        elif _== stop_state_log:
            logger.plot_states()
        elif i == stop_state_log:
            logger.plot_states()

        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()

    # 确保正确地关闭视频文件，以保存录制的内容
    if RENDER:
        video.release()

if __name__ == '__main__':
    # 是否将策略导出为jit模块（用于从C++运行它）
    EXPORT_POLICY = False
    # 是否进行设置相机、录制视频、渲染环境等
    RENDER = False
    # 是使用预设的速度，还是使用实时计算的速度值
    FIX_COMMAND = False
    # 创建超参数实例
    args = get_args()
    # 开始运行策略
    play(args)