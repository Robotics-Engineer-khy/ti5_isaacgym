import torch
import torch.nn as nn
import torch.optim as optim
# print(f"Actor MLP: {self.actor}")
# print(f"Critic MLP: {self.critic}")
# print(f"long_history CNN: {self.long_history}")
# print(f"state_estimator MLP: {self.state_estimator}")
from .actor_critic_dh import ActorCriticDH
from .rollout_storage import RolloutStorage


# 创建DHPPO实例化对象，actor_critic=ActorCriticDH，device = cuda:0
class DHPPO:
    actor_critic: ActorCriticDH
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 lin_vel_idx = 45,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic: ActorCriticDH = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(),lr=learning_rate)
        self.state_estimator_optimizer = optim.Adam(self.actor_critic.state_estimator.parameters(),lr=learning_rate)

        # 创建一个新的Transition实例，用于存储当前时间步（或当前交互周期）中智能体与环境交互产生的数据。
        # Transition是一个简单的数据容器，通常用于表示一个强化学习交互的单个时间步的数据
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        # 1
        self.num_learning_epochs = num_learning_epochs
        # 1
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_short_obs = self.actor_critic.num_short_obs
        self.lin_vel_idx = lin_vel_idx

    # 是初始化RolloutStorage数据结构，用于存储强化学习过程中多个环境的交互数据
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # RolloutStorage是一个用于存储轨迹数据（如观察、动作、奖励等）的容器，并允许在训练过程中批量提取这些数据进行学习
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, None, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        # 调用actor_critic网络的act方法，基于obs（当前环境观察）生成动作，.detach()：这个方法用于从计算图中分离出该动作
        self.transition.actions = self.actor_critic.act(obs).detach()
        # 调用actor_critic的evaluate()方法，基于另一个观察critic_obs计算当前状态的值（value）
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        # 计算并返回给定动作 actions 在当前策略下的 对数概率（log probability）。这是强化学习中策略梯度方法中的一个重要步骤，通常用于计算与动作选择相关的损失或奖励。
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        #  返回了当前 正态分布的均值，也就是策略网络根据观察数据计算得到的期望动作。
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        # 返回了当前 正态分布的标准差，即 stddev，它表示当前策略下动作分布的 离散程度（或不确定性）。
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        # 返回基于当前环境生成的动作
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    # 计算每个时间步的回报和优势
    def compute_returns(self, last_critic_obs):
        # 获得状态估计价值
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        # 计算每个时间步的回报和优势
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_state_estimator_loss = 0

        # 通过生成器（mini_batch_generator）迭代地从存储中获取小批次数据，用于训练。每次训练会进行多次更新（num_learning_epochs）和多次小批次训练（num_mini_batches）
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                # 使用当前批次的数据来执行一个动作
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                # 获取状态估计器的输入并计算预测的线性速度
                state_estimator_input = obs_batch[:,-self.num_short_obs:]
                est_lin_vel = self.actor_critic.state_estimator(state_estimator_input)
                ref_lin_vel = critic_obs_batch[:,self.lin_vel_idx:self.lin_vel_idx+3].clone()
                # 获取当前批次的动作日志概率
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # 通过Critic计算当前状态的价值
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                # 获取当前的均值(mu)和标准差(sigma)，以及熵值
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                # 计算KL散度，用于自适应学习率调节
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                # 计算Surrogate损失
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # 综合所有损失计算总损失
                loss = (surrogate_loss + 
                        self.value_loss_coef * value_loss - 
                        self.entropy_coef * entropy_batch.mean() +
                        torch.nn.MSELoss()(est_lin_vel, ref_lin_vel))
                
                # Gradient step
                # 反向传播并更新参数
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 计算状态估计器损失
                state_estimator_loss = torch.nn.MSELoss()(est_lin_vel, ref_lin_vel)
                # self.state_estimator_optimizer.zero_grad()
                # state_estimator_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.state_estimator.parameters(), self.max_grad_norm)
                # self.state_estimator_optimizer.step()

                # 累积各项损失的均值
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_state_estimator_loss += state_estimator_loss.item()

        # 计算更新的总次数
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_state_estimator_loss /= num_updates
        # 清空存储
        self.storage.clear()

        # 返回价值函数损失（value loss）、策略损失（surrogate loss）和状态估计损失（state estimator loss）
        return mean_value_loss, mean_surrogate_loss, mean_state_estimator_loss
