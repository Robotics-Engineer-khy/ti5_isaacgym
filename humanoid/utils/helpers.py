import os
import copy
import torch
import numpy as np
import random
# （62）Importing module 'gym_38' (/home/khy/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
# （92）Setting GYM_USD_PLUG_INFO_PATH to /home/khy/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
from isaacgym import gymapi
from isaacgym import gymutil

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# 将一个对象（类实例或类本身）递归地转换为一个字典
def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    # 设置Python随机模块的种子
    random.seed(seed)
    # 设置NumPy随机模块的种子
    np.random.seed(seed)
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 设置Python哈希种子
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 设置PyTorch的GPU随机种子
    torch.cuda.manual_seed(seed)
    # 设置所有GPU（如果有多个 GPU）的随机种子
    torch.cuda.manual_seed_all(seed)

    # For cudnn backend to ensure reproducibility
    # 设置cuDNN后端的确定性
    torch.backends.cudnn.deterministic = True
    # 禁用cuDNN优化
    torch.backends.cudnn.benchmark = False

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # 初始化仿真参数sim_params，SimParams类的实例化，在gym_38.so静态库中
    sim_params = gymapi.SimParams()
    # 为args参数设置一些值
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    # 物理引秦默认为 gymapi.SIM_PHYSX,值为True
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # True
        sim_params.physx.use_gpu = args.use_gpu
        # o
        sim_params.physx.num_subscenes = args.subscenes
    # True
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    # if sim options are provided in cfg, parse them and update/override above:
    # True
    if "sim" in cfg:
        # 该函数的作用是从给定的配置字典cfg中解析仿真参数，并更新传入的sim_params对象（sim_options）。
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    # 返回解析更新后的sim_params
    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        # 列出root目录下的所有文件和子目录，并返回一个列表runs
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        # 将runs列表中的目录按字典顺序（通常是字母顺序）排序
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        # 选择列表中最后一个元素，通常是最新的训练运行目录（因为它已经排过序）
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        # 代码将查找load_run目录中包含 "model" 字样的所有文件
        models = [file for file in os.listdir(load_run) if "model" in file]
        # 对模型文件名进行排序
        models.sort(key=lambda m: "{0:0>15}".format(m))
        # 选择排序后的最后一个模型文件（通常是最新的模型）
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {   "name": "--task",
            "type": str,
            "default": "t1_dh_stand",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",},
        {   "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",},
        {   "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",},
        {   "name": "--run_name",
            "type": str,
            "default": "ti5",
            "help": "Name of the run. Overrides config file if provided.",},
        {   "name": "--load_run",
            "type": str,
            # "default": -1,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",},
        {   "name": "--checkpoint",
            "type": int,
            # "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",},
        {   "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",},
        {   "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",},
        {   "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",},
        {   "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",},
        {   "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",},
        {   "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",},]

    # 解析参数
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    # 名称统一
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args

# 用于将一个强化学习模型的策略网络（Actor）导出为一个TorchScript格式的模型,可以在不同的设备上（如 C++ 环境）进行部署
def export_policy_as_jit(actor_critic, path):
    # 创建保存路径，若路径已存在则不报错
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.pt")
    # 复制并移动模型到CPU
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    # 将模型转换为TorchScript
    traced_script_module = torch.jit.script(model)
    # 将TorchScript模型保存到指定路径
    traced_script_module.save(path)
