from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from record_config import record_config

def train(args):
    # env是创建的T1DHStandEnv环境对象，env_cfg是返回环境参数的DHT1StandCfg对象
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # 创建DHOnPolicyRunner对象、DHT1StandCfgPPO()对象、logs存储路径
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # 记录与双足机器人模拟相关的配置文件,都保存在'logs/t1_dh_stand/exported_data/当前时间'文件中
    record_config(log_root=log_dir,urdf_path=env_cfg.asset.file,name=args.task)
    # 开始训练，训练次数max_iterations轮
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    # 创建超参数实例
    args = get_args()
    # 显示图形渲染界面（False）
    args.headless = False
    # 是否继续训练
    args.resume = True
    # 开始训练超参数对象
    train(args)
