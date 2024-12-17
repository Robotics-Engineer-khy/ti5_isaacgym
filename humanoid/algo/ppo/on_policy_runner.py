import os
import time
import torch
import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo import PPO
from .actor_critic import ActorCritic
from humanoid.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        # print('1')
        # print(self.env.num_obs)
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.it = 0

        _, _ = self.env.reset()

    # 传入最大迭代次数、是否初始化随即ep长度（站立训练时为Fasle）
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        # 检查是否需要初始化日志和写入器
        if self.log_dir is not None and self.writer is None:
            # wandb.init(
            #     project="XBot",
            #     sync_tensorboard=True,
            #     name=self.wandb_run_name,
            #     mode="disabled",
            #     config=self.all_cfg,
            # )
            # 初始化writer，用于将训练过程中产生的日志信息写入到TensorBoard中，用于可视化训练过程中的各种数据（如损失值、指标、图像等）。
            # 每10s刷新一次
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        # 训练时运行（站立时不运行），生成一个随机的episode长度，整数的值范围是从0到self.env.max_episode_length-1，即0到2399
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # 获取观察空间，用于actor观察值
        # 获取一个行数为4096，列数为3102的二维全零张量
        obs = self.env.get_observations()
        # 获取一个行数为4096，列数为219的二维全零张量
        privileged_obs = self.env.get_privileged_observations()
        # 用于critic的观察值
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # 确保obs和critic_obs这两个张量转移到指定的设备GPU上
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        # 将神经网络设置为训练模式，会启用训练特性，如随机失活（dropout），用于防止过拟合，或者批量归一化（batch normalization），用于加速训练。
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        # 设定奖励缓冲区，最多保存100个元素的奖励缓冲区
        rewbuffer = deque(maxlen=100)
        # 设定回合长度缓冲区，最多保存100个元素的回合长度缓冲区
        lenbuffer = deque(maxlen=100)
        # 初始化当前奖励综合，创建一个4096的一维全零张量
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # 初始化当前回合长度，创建一个4096的一维全零张量
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # 总迭代次数 = 0（当前学习迭代次数） + 20000（将要执行迭代次数）
        tot_iter = self.current_learning_iteration + num_learning_iterations
        # 从当前学习迭代次数到总迭代次数（左闭又开）
        for it in range(self.current_learning_iteration, tot_iter):
            self.it = it
            # 每次迭代开始记录时间戳
            start = time.time()
            # Rollout
            # 当你进入推理模式时，PyTorch 会关闭一些不必要的操作（如梯度计算），从而减少内存使用并提高性能。
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 通过actor的的观察值，计算出动作action，通过critic的观察值计算出值value
                    actions = self.alg.act(obs, critic_obs)
                    # 返回actor的观察，critic的观察值，奖励值，重置，额外信息
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    # 将信息上传到GPU上
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),)
                    # 处理在强化学习环境中每个时间步的结果，包括奖励、结束标志、额外信息等
                    self.alg.process_env_step(rewards, dones, infos)

                    # 处理环境步骤，包括执行动作、计算奖励、更新奖励总和和回合长度缓冲区等。
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        # 奖励累积
                        cur_reward_sum += rewards
                        # 当前episode长度加1，每个环境都加1
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # 将选定环境的累计奖励添加到奖励缓冲区
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        # 将选定环境的episode步数添加到步数缓冲区
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        # 表示将new_ids中对应的所有智能体或回合的奖励总和重置为0
                        cur_reward_sum[new_ids] = 0
                        # 表示将new_ids中对应的所有智能体或回合的步数/长度重置为0
                        cur_episode_length[new_ids] = 0
                # 记录停止时间戳
                stop = time.time()
                # 记录收集数据的时间
                collection_time = stop - start

                # Learning step
                # 计算回报，并执行更新步骤。
                start = stop
                # 计算奖励值
                self.alg.compute_returns(critic_obs)

            # 表示在推理模式结束

            # 记录更新时间和损失，如果设置了日志目录，则记录日志并根据保存间隔保存模型。
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            # 记录更新学习时间
            learn_time = stop - start

            if self.log_dir is not None:
                # 记录当前作用域中的局部变量到日志文件中。
                self.log(locals())
            # 每100轮训练，就保存一次模型
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            # 清空
            ep_infos.clear()
        # 表示一轮迭代结束

        # 训练结束后，当前迭代次数变成总训练次数
        self.current_learning_iteration += num_learning_iterations
        # 保存最后一次训练模型
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()

        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                # 计算速度
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # 价值损失函数
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                # 代理损失
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                # 动作噪声标准差
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                # 平均奖励
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                # 平均回合长度
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            # 总时间步长
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            # 迭代时间
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            # 总时间
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            # 预计剩余时间（ETA），基于平均每次迭代所需时间，估计剩余的训练时间
            f"""{'Estimated Time Of Arrival:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                # 这个键保存了模型的状态字典（state_dict）。在强化学习中，通常包含了神经网络（如策略网络、值网络等）的权重和偏置。
                "model_state_dict": self.alg.actor_critic.state_dict(),
                # 保存优化器的状态字典。self.alg.optimizer.state_dict() 返回当前优化器的状态，包括已学习的动量、梯度等信息。保存优化器的状态是非常重要的，这样在恢复训练时可以从断点继续，而不需要重新初始化优化器的状态。
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                # 保存当前的迭代次数（或训练进度）。self.it 代表当前的迭代计数。这样在恢复训练时，你可以知道从哪个迭代开始继续训练，而不需要从头开始。
                "iter": self.it,
                # 保存附加的额外信息。
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        # 将神经网络设置为评估模式，当模型在训练模式下时，某些层（如dropout和batch normalization）会有不同的行为，而在评估模式下，模型会禁用这些行为，以保证模型的预测稳定性。
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            # 将神经网络传到指定GPU上
            self.alg.actor_critic.to(device)
        # 根据给定的观察数据通过actor网络计算出动作的均值，该动作均值是智能体在该状态下的期望行为
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
