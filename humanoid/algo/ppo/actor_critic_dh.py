import torch
import torch.nn as nn
from torch.distributions import Normal


# 创建ActorCriticDH实例化对象，num_short_obs = 235，num_proprio_obs = 47，num_critic_obs = 219，num_actions = 12，对应DHT1StandCfgPPO中的policy类
class ActorCriticDH(nn.Module):
    def __init__(self,  num_short_obs,
                        num_proprio_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        state_estimator_hidden_dims=[256, 128, 64],
                        in_channels = 66,
                        kernel_size=[6, 4],# 卷积核大小
                        filter_size=[32, 16],# 每一层卷积的输出通道数
                        stride_size=[3, 2],# 步幅大小
                        lh_output_dim=64,
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCriticDH.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDH, self).__init__()

        
        # define actor net and critic net
        # 5 * 47 + 64 +3 = 302

        mlp_input_dim_a = num_short_obs + lh_output_dim + 3
        # 66 * 47 + 64 + 3 = 3102
        # mlp_input_dim_a = num_short_obs
        # print(mlp_input_dim_a)
        # 输入219
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []

        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # print(actor_hidden_dims[0])
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)

        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        #
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        #define long_history CNN
        long_history_layers = []
        # 66
        self.in_channels = in_channels
        # 47
        cnn_output_dim = num_proprio_obs
        # 卷积的输出通道数、卷积核大小、步幅大小
        # [32,6,3],[16,4,2]
        for out_channels, kernel_size, stride_size in zip(filter_size, kernel_size, stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            # （47 - 6 + 3）// 3 = 14
            # (14 - 4 + 2) // 2 = 6
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            in_channels = out_channels
        # 96 = 6 * 16
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim, 128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*long_history_layers)
        print(f"long_history CNN: {self.long_history}")

        # define state_estimator MLP
        # 235 = 47 * 5
        self.num_short_obs = num_short_obs
        # 网络输入为235，输出为256
        state_estimator_input_dim = num_short_obs
        state_estimator_output_dim = 3
        state_estimator_layers = []
        state_estimator_layers.append(nn.Linear(state_estimator_input_dim, state_estimator_hidden_dims[0]))
        state_estimator_layers.append(activation)
        for l in range(len(state_estimator_hidden_dims)):
            if l == len(state_estimator_hidden_dims) - 1:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l], state_estimator_output_dim))
            else:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l], state_estimator_hidden_dims[l + 1]))
                state_estimator_layers.append(activation)

        # 将多个神经网络层（state_estimator_layers中的层）按照顺序堆叠在一起，形成一个新的神经网络
        # 这个神经网络负责处理输入数据（例如环境的短期历史观察），并生成状态估计值
        self.state_estimator = nn.Sequential(*state_estimator_layers)

        # 状态估计器网络
        print(f"state_estimator MLP: {self.state_estimator}")
        # 47
        self.num_proprio_obs = num_proprio_obs
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # 使用了一个神经网络（self.actor）来生成概率分布的均值（mean），并且通过设定一个固定的标准差（self.std）来定义这个分布
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        # 从输入的observations中提取出最近的一段短期历史，5帧 * 47 = 235
        short_history = observations[...,-self.num_short_obs:]
        # 5帧用来做状态评估
        es_vel = self.state_estimator(short_history)
        # 对observations数据进行形状调整（view()）后，输入到long_history网络，生成压缩后的长期历史信息。
        # observations是一个原始形状为[4096,66,47]，转换为张量，输入到网络
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        # 将short_history、es_vel 和 compressed_long_history在最后一个维度上拼接起来，生成一个包含所有相关信息的观察（actor_obs）
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history),dim=-1)
        # 根据拼接后的观察（actor_obs），更新策略网络的分布（distribution）。
        self.update_distribution(actor_obs)
        # 从正态分布中采样一个值，表示智能体在当前状态下的动作
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 根据给定的观察数据通过actor网络计算出动作的均值，该动作均值是智能体在该状态下的期望行为
    def act_inference(self, observations):
        # 从输入的observations中提取出最近的一段短期历史，5帧 * 47 = 235
        short_history = observations[...,-self.num_short_obs:]
        # 5帧用来做状态评估，输出为3
        es_vel = self.state_estimator(short_history)
        # 对observations数据进行形状调整（view()）后，输入到long_history网络，生成压缩后的长期历史信息。
        # observations是一个原始形状为[4096,66,47]，转换为张量，输入到网络
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        # 将short_history、es_vel 和 compressed_long_history在最后一个维度上拼接起来，生成一个包含所有相关信息的观察（actor_obs）
        # 235 + 3 + 64 = 302
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history),dim=-1)
        # 输入302，输出12，表示期望动作
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # critic神经网络根据观察空间，输出状态估计价值
        value = self.critic(critic_observations)
        return value