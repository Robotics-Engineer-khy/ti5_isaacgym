import numpy as np
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DHT1StandCfg(LeggedRobotCfg):
    """
    Configuration class for the ti5 humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # 连续帧，智能体利用连续帧的历史信息，能够更好地理解环境的动态变化，提高决策的准确性
        frame_stack = 66      # long history step
        short_frame_stack = 5   # short history step
        c_frame_stack = 3

        # 单个时间步的观察数据的维度
        num_single_obs = 47     # without bool value for stand
        # num_single_obs = 42     # add bool value for stand
        # 观察空间的总维度 3102 = 66 * 47
        num_observations = int(frame_stack * num_single_obs)
        # 单个特权观察维度
        single_num_privileged_obs = 73

        # 特权观察空间总维度219 = 3 * 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        # 动作空间维度
        num_actions = 12
        # 环境总数量
        num_envs = 4096
        # 每个回合时间最长为24s（以秒为单位）
        episode_length_s = 24
        # 是否使用期望动作
        use_ref_actions = False
        single_linvel_index = 53
        num_commands = 5

    class safety:
        # 位置限制系数 * urdf位置lower/upper = 位置限制
        pos_limit = 1.0
        # 速度限制系数 * urdf速度velocity = 速度限制
        vel_limit = 1.0
        # 扭矩限制系数 * urdf力矩effort = 扭矩限制
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        # 绝对路径/home/khy/standing/resources/robots/t1/urdf/t1.urdf
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/t1/urdf/t1.urdf'
        name = "t1"
        foot_name = "6_link"
        knee_name = "4_link"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        '''
        # heightfield的属性设置
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        '''
        # 地形有'plane','heightfield','trimesh'
        mesh_type = 'trimesh'
        # 课程学习的核心思想是逐步增加任务的难度，而不是一开始就让智能体面对极其困难的环境或任务。通过逐步增加难度，智能体能够更快地收敛，避免陷入局部最优解或长时间未能学习到有效策略。
        curriculum = True
        # rough terrain only:
        measure_heights = False
        # 设置地面平面静态摩擦力（即物体静止时的摩擦力）
        static_friction = 0.6
        # 设置地面平面的动态摩擦力（即物体滑动时的摩擦力）
        dynamic_friction = 0.6
        terrain_length = 8
        terrain_width = 8
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3
        terrain_dict = {"flat": 0.5, 
                        "rough flat": 0.3,
                        "slope up": 0.1,
                        "slope down": 0.1, 
                        "rough slope up": 0,
                        "rough slope down": 0,
                        "stairs up": 0,
                        "stairs down": 0,
                        "discrete": 0,
                        "wave": 0,}

        terrain_proportions = list(terrain_dict.values())
        # 地形为'heightfield', 'trimesh'时，需要设置的参数
        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]   # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        # 设置地面平面的恢复系数（即碰撞后物体反弹的弹性，0表示完全无弹性，1表示完全弹性）
        restitution = 0

    # 添加噪声扰动
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            # 关节位置
            dof_pos = 0.02
            # 关节速度
            dof_vel = 1.5
            # 角速度
            ang_vel = 0.2
            # 线速度
            lin_vel = 0.1
            # 姿态
            quat = 0.1
            # 重力
            gravity = 0.05
            # 高度
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        # 初始高度
        pos = [0.0, 0.0, 1.1]
        # 关节初始位置(17度左右)
        init_angle = 0.3
        # 初始姿态的默认关节角度（弧度制）
        default_joint_angles = {
            'leg_l1_joint': 0,
            'leg_l2_joint': 0,
            'leg_l3_joint': -init_angle,
            'leg_l4_joint': init_angle * 2,
            'leg_l5_joint': -init_angle,
            'leg_l6_joint': 0,
            'leg_r1_joint': 0,
            'leg_r2_joint': 0,
            'leg_r3_joint': -init_angle,
            'leg_r4_joint': init_angle * 2,
            'leg_r5_joint': -init_angle, 
            'leg_r6_joint': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'1_joint': 50 * 1, '2_joint': 70 * 1, '3_joint': 90 * 1,'4_joint': 120 * 1,
                     '5_joint': 50 * 1, '6_joint': 30 * 1}
        damping = {'1_joint': 5, '2_joint': 7, '3_joint': 9,'4_joint': 12,
                     '5_joint': 5, '6_joint': 3}

        # action scale: target angle = action_scale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    # gym平台的一些内部参数
    class sim(LeggedRobotCfg.sim):
        # 步进器
        dt = 0.001  # 1000Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    # 域随机化
    class domain_rand(LeggedRobotCfg.domain_rand):
        # 摩擦力随机化
        randomize_friction = True
        # 摩擦力系数随机化范围
        friction_range = [0.2, 1.3]
        # 恢复系数较低，表示碰撞后物体会有一定的弹性反弹，但弹性较弱（通常意味着碰撞后速度会有显著减少）
        restitution_range = [0.0, 0.4]

        # 推力随机化
        push_robots = False
        # 推力时间间隔
        push_interval_s = 6
        update_step = 2500*24
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        # 外力（推力和扭矩）随机化
        add_ext_force = True
        ext_force_max_x = 600
        ext_force_max_y = 400
        ext_force_max_z = 5
        ext_torque_max = 0
        ext_force_interval_s = 4
        add_update_step = 4000 * 24
        add_duration = [0.0, 0.05, 0.1, 0.15]

        # 基座质量随机化
        randomize_base_mass = True
        # 基座质量 +- 该参数
        added_mass_range = [-2.5, 2.5]

        # 质心随机化
        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],[-0.05, 0.05],[-0.05, 0.05]]

        # pd随机化
        randomize_gains = True
        # kp和kd的值 * 该系数
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        # 扭矩随机化
        randomize_torque = True
        # 扭矩 * 该系数
        torque_multiplier_range = [0.8, 1.2]

        # 连杆质量随机化
        randomize_link_mass = True
        # 连杆质量 * 该系数
        added_link_mass_range = [0.9, 1.1]

        # 位置偏移随机化
        randomize_motor_offset = True
        # 位置 +- 该参数
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        '''
        # 关节摩擦力随机化
        randomize_joint_friction = False
        # 每个关节摩擦力随机化
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]   #multiplier
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        # 关节阻尼随机化，阻尼用于模拟如摩擦力、空气阻力、弹簧-阻尼系统等系统的能量耗散
        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]       #multiplier
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]
        '''

        # armature可以表示电机系统的惯性或与驱动电机有关的控制参数，例如电机的反作用力矩或电机的电磁阻力。
        randomize_joint_armature = True
        randomize_joint_armature_each_joint = True
        joint_armature_range = [0.001, 0.05]     # Factor
        joint_1_armature_range = [0.15 *0.8, 0.15 *1.2]
        joint_2_armature_range = [0.15 *0.8, 0.15*1.2]
        joint_3_armature_range = [3.6 *0.5, 3.6*1.0]
        joint_4_armature_range = [3.6 *0.5, 3.6*1.0]
        joint_5_armature_range = [0.1 *0.5, 0.1*1.1]
        joint_6_armature_range = [0.028 *0.5, 0.028*1.5]

        joint_7_armature_range = [0.15 *0.8, 0.15*1.2]
        joint_8_armature_range = [0.15 *0.8, 0.15*1.2]
        joint_9_armature_range = [3.6 *0.5, 3.6*1.0]
        joint_10_armature_range = [3.6 *0.5, 3.6*1.0]
        joint_11_armature_range = [0.1 *0.5, 0.1*1.1]
        joint_12_armature_range = [0.028 *0.5, 0.028*1.5]

        # 延迟随机化
        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        # 是否在每个环境步骤中随机化延迟的时间步数范围
        lag_timesteps_range = [0,30]

        # 接受信号的延迟随机化(dof_pos和dof_vel),dof_pos和dof_vel延迟一样
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False  # 不常用always False
        dof_lag_timesteps_range = [0, 30]

        # 接收信号的延迟随机化(dof_pos和dof_vel),dof_pos和dof_vel延迟不同
        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = True
        randomize_dof_pos_lag_timesteps_perstep = False          # 不常用always False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = True
        randomize_dof_vel_lag_timesteps_perstep = False          # 不常用always False
        dof_vel_lag_timesteps_range = [7, 25]

        # imu延迟随机化
        add_imu_lag = True
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False         # 不常用always False
        imu_lag_timesteps_range = [0, 10]

        # 库伦摩擦力随机化
        randomize_coulomb_friction = True
        # 扭矩 +- 该参数（库伦力的大小）
        joint_coulomb_range = [0.1, 1.0]
        # 速度 * 该系数（粘性摩擦力）
        joint_viscous_range = [0.1, 0.9]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 25  # time before command are changed[s]
        gait = ["walk_omnidirectional","stand","walk_omnidirectional"]
        # gait = ["stand"]
        # gait_time_range = [2,6]
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,3],
                           "stand": [2,3],
                           "walk_omnidirectional": [4,6]}
        stand_time = 18  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        # 站立姿态下质心的阈值
        stand_com_threshold = 0.05
        sw_switch = True

        # 每个方向的速度的范围
        class ranges:
            # x轴线速度范围
            lin_vel_x = [-0.5, 0.5]     # min max [m/s]
            # lin_vel_x = [-0.5, 1]     # min max [m/s]
            # y轴线速度范围
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            # lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            # 偏航角速度范围
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            # 前进方向的范围
            heading = [-3.14, 3.14]

    class rewards:
        # 基座期望高度
        base_height_target = 0.965
        # 脚间最小欧几里德距离
        foot_min_dist = 0.15
        # 脚间最大欧几里德距离
        foot_max_dist = 0.45
        # 膝间最小欧几里德距离
        knee_min_dist = 0.12
        # 膝间最大欧几里德距离
        knee_max_dist = 0.35

        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.3    # rad joint_angle_limit = default_joint_angles +- target_joint_pos_scale
        # 脚抬高最低距离
        target_feet_height = 0.02       # m
        # 脚抬高的最高距离
        target_feet_height_max = 0.08
        # 脚摆动周期
        cycle_time = 0.8                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 只有当奖励为正值的时候，奖励才会被累计（防止开始提前终止）
        only_positive_rewards = True
        # tracking reward = exp(error*sig75ma)
        tracking_sigma = 5 
        max_contact_force = 500

        # 每个奖励函数的返回值 * 缩放因子 = 累积奖励值
        class scales:
            joint_pos = 4
            feet_clearance = 1
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1
            foot_slip = -0.5
            feet_distance = 0.2
            knee_distance = 0.2
            feet_rotation = 0.8
            # contact 
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.8
            vel_mismatch_exp = 0.5 
            low_speed = 0.2
            track_vel_hard = 0.5

            # base pos
            default_joint_pos = 1
            orientation = 1
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.03
            torques = -2e-7
            dof_vel = -2e-5
            dof_acc = -5e-7
            collision = -1
            stand_still = 2.5
            # stand_sysmetry = 1.0

    class normalization:
        # 观察空间缩放系数
        class obs_scales:
            lin_vel = 2
            ang_vel = 1
            dof_pos = 1
            dof_vel = 0.05
            quat = 1
            height_measurements = 5.0
        clip_observations = 100
        # 动作裁减范围step时会用到
        clip_actions = 100

class DHT1StandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    # train时make_alg_runner需要创建一个名字为DHOnPolicyRunner的类，play时需要打印出来
    runner_class_name = 'DHOnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]

        # for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   # long history output dim
        in_channels = DHT1StandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # 熵系数
        entropy_coef = 0.001
        # 学习率
        learning_rate = 1e-5
        # 学习的总轮数
        num_learning_epochs = 2
        # 折扣因子（用来计算未来奖励的衰减）
        gamma = 0.994
        # GAE（Generalized Advantage Estimation）中的lambda参数
        lam = 0.9
        # 每次训练中使用的小批量数量
        num_mini_batches = 4
        if DHT1StandCfg.terrain.measure_heights:
            lin_vel_idx = (DHT1StandCfg.env.single_num_privileged_obs + DHT1StandCfg.terrain.num_height) * (DHT1StandCfg.env.c_frame_stack - 1) + DHT1StandCfg.env.single_linvel_index
            # push_frc_idx = (DHT1StandCfg.env.single_num_privileged_obs + DHT1StandCfg.terrain.num_height) * (DHT1StandCfg.env.c_frame_stack - 1) + DHT1StandCfg.env.single_linvel_index + 9
        else:
            # 73 * （3 - 1）+ 53 = 199
            lin_vel_idx = DHT1StandCfg.env.single_num_privileged_obs * (DHT1StandCfg.env.c_frame_stack - 1) + DHT1StandCfg.env.single_linvel_index
            # push_frc_idx = DHT1StandCfg.env.single_num_privileged_obs * (DHT1StandCfg.env.c_frame_stack - 1) + DHT1StandCfg.env.single_linvel_index + 9
        # est_push_state_bool = False

    class runner:
        # train时，实例化DHOnPolicyRunner类时，需要创建一个名字为ActorCriticDH的类
        policy_class_name = 'ActorCriticDH'
        # train时，实例化DHOnPolicyRunner类时，需要创建一个名字为DHPPO的类
        algorithm_class_name = 'DHPPO'

        # 每次迭代，每个环境交互的步数
        num_steps_per_env = 24  # per iteration
        # 总训练次数
        max_iterations = 30000  # number of policy updates

        # 模型保存的间隔，即多少次迭代更新一次
        save_interval = 500
        experiment_name = 't1_dh_stand'
        # 渲染或录制视频会用到
        run_name = 'ti5'

        # 是否继续训练
        resume = False
        # 表示加载策略的绝对路经
        # load_run = '/home/khy/biped_rl_ti5_deliver/logs/t1_dh_stand/exported_data/2024-10-29_16-01-17'
        # load_run = '2024-10-29_16-01-17'
        load_run = -1  # -1 = last run
        # 表示所选路径的最后保存的一个模型
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt