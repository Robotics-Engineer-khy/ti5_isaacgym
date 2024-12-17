import torch
from typing import Tuple, Union
from abc import ABC, abstractmethod

# VecEnv类是一个抽象基类，提供了多环境管理的统一接口，它指定了一个最小化的接口，用于在强化学习（RL）设置中实现多环境
class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_short_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor 
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor # current episode duration
    extras: dict
    device: torch.device
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass