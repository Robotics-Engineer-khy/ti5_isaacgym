import os
from typing import Tuple
from datetime import datetime
# print(f"Actor MLP: {self.actor}")
# print(f"Critic MLP: {self.critic}")
# print(f"long_history CNN: {self.long_history}")
# print(f"state_estimator MLP: {self.state_estimator}")
from humanoid.algo import VecEnv
from humanoid.algo import OnPolicyRunner, DHOnPolicyRunner

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
# 运行register函数
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    # 注册分别对应T1DHStandEnv, DHT1StandCfg(), DHT1StandCfgPPO()三类
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    # 返回环境参数和训练参数，返回类型是一个元组，包含两个元素
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        env_cfg = self.env_cfgs[name]# DHT1StandCfg()环境参数
        train_cfg = self.train_cfgs[name]# DHT1StandCfgPPO()训练参数
        env_cfg.seed = train_cfg.seed# 5
        return env_cfg, train_cfg

    # 在已注册和提供的配置文件中，创建一个环境
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered name or from the provided config file.
        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.
        Raises:
            ValueError: Error if no registered env corresponds to 'name'
        Returns:
            isaac gym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        if args is None:
            args = get_args()

        # train时name为默认值"t1_dh_stand"
        if name in self.task_classes:
            # task_class为T1DHStandEnv类
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        # train时为None
        if env_cfg is None:
            # 返回t1_dh_stand_config.py的环境参数和训练参数
            env_cfg, _ = self.get_cfgs(name)
        # 更新参数（训练时不更新）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # 设置随机种子
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        # 返回键为"sim"，值为sim的属性及其内部physx组成的字典
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        # 解析参数并更新参数到仿真器上，sim_params为SimParams类的实例，它存储了仿真环境的各种配置参数
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,# DHT1StandCfg()环境参数
                            sim_params=sim_params,# 仿真环境的各种配置参数
                            physics_engine=args.physics_engine,# SimType.SIM_PHYSX
                            sim_device=args.sim_device,#"cuda:0"
                            headless=args.headless)# False
        self.env_cfg_for_wandb = env_cfg
        return env, env_cfg

    # 创建训练算法，返回创建OnPolicyRunner对象和LeggedRobotCfgPPO对象
    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.
        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.
        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored
        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """

        # train时args不为None
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        # train时train_cfg为None，name为"t1_dh_stand"
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 返回t1_dh_stand_config.py的DHT1StandCfg()环境参数和DHT1StandCfgPPO()训练参数
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # 如果参数特殊，在args中有默认值，则更新参数
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        # 当前时间
        current_date_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # train时为default，创建logs路径
        if log_root=="default":
            # /home/khy/standing/logs/t1_dh_stand/exported_data
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_data')
            # /home/khy/standing/logs/t1_dh_stand/exported_data/2024-当前时间
            log_dir = os.path.join(log_root, current_date_time_str + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, current_date_time_str + train_cfg.runner.run_name)

        # 最终的all_cfg将包含来自两个字典的所有键值对，且相同的键只保留env_cfg_dict中的值
        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)
        all_cfg = {**train_cfg_dict, **env_cfg_dict}

        # 创建一个名字为DHOnPolicyRunner的类
        runner_class = eval(train_cfg_dict["runner_class_name"])
        # 创建DHOnPolicyRunner的类的实例化，传入参数为T1DHStandEnv，DHT1StandCfg()环境参数和DHT1StandCfgPPO()训练参数，模型所在路径，cuda:0
        runner = runner_class(env, all_cfg, log_dir, device=args.rl_device)

        # True表示加载之前训练好的模型，是否继续训练
        resume = train_cfg.runner.resume
        if resume:
            # 获取加载训练模型的路径
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            # 加载模型的状态、优化器等信息
            runner.load(resume_path, load_optimizer=False)

        return runner, train_cfg, log_dir

# make global task registry
task_registry = TaskRegistry()