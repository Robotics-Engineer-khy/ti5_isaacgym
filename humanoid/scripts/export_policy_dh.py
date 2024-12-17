from humanoid import LEGGED_GYM_ROOT_DIR
import os
import copy

from humanoid.envs import *
from humanoid.utils import  get_args, task_registry, Logger
from humanoid.utils.helpers import class_to_dict
from datetime import datetime
from extract_config import extract_parameters_from_file, save_parameters_to_yaml
import torch
from humanoid.algo.ppo import ActorCriticDH
import os

class ExportedDH(torch.nn.Module):
    def __init__(self, actor, long_history, state_estimator, num_short_obs, in_channels, num_proprio_obs):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.long_history = copy.deepcopy(long_history).cpu()
        self.state_estimator = copy.deepcopy(state_estimator).cpu()
        self.num_short_obs = num_short_obs
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs
    
    def forward(self, observations):
        short_history = observations[...,-self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history),dim=-1)
        actions_mean = self.actor(actor_obs)
        # return actions_mean
        return actions_mean, es_vel
    
    def export(self, path):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)  
    
def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path, load_run
def export_policy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    num_critic_obs = env_cfg.env.num_privileged_obs
    if env_cfg.terrain.measure_heights:
        num_critic_obs = env_cfg.env.c_frame_stack * (env_cfg.env.single_num_privileged_obs +env_cfg.terrain.num_height)
    num_short_obs = env_cfg.env.short_frame_stack * env_cfg.env.num_single_obs
    actor_critic_class = eval(train_cfg_dict["runner"]["policy_class_name"])  # ActorCriticDH
    actor_critic: ActorCriticDH = actor_critic_class(
        num_short_obs, env_cfg.env.num_single_obs, num_critic_obs, env_cfg.env.num_actions, **policy_cfg
    )
    # load policy
    log_root_encoder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_data')
    model_path, load_run = get_load_path(log_root_encoder, load_run=args.load_run, checkpoint=args.checkpoint)
    print("Load model from:", model_path)
    loaded_dict = torch.load(model_path)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    
    exported_policy = ExportedDH(actor_critic.actor,
                                 actor_critic.long_history,
                                 actor_critic.state_estimator,
                                 num_short_obs,
                                 policy_cfg["in_channels"],
                                 env_cfg.env.num_single_obs)
    
    # export policy as a jit module (used to run it from C++)
    current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_policies',
                            current_date_time)
    os.makedirs(root_path, exist_ok=True)
    dir_name = "policy_dh.jit"
    path = os.path.join(root_path, dir_name)
    exported_policy.export(path)
    print("Export policy to:", path)
    
    return load_run

def export_config(load_run, export_cloud=False):
    if(export_cloud):
        load_run = '/home/bridge/A_rl/biped_rl_ti5/logs/t1_dh_stand/exported_data/111cloud'
   
    output_file = '/home/bridge/A_rl/biped_rl_ti5/logs/t1_dh_stand/exported_yaml' + "/ti5_amp.yaml"

    # 提取参数
    parameters = extract_parameters_from_file(load_run+"/t1_dh_stand_config.txt")

    # 保存参数到YAML文件
    save_parameters_to_yaml(parameters, output_file)

    print("参数已成功提取并保存到"+output_file+"文件中。")
    
if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    if args.load_run == None:
        args.load_run = -1
    if args.checkpoint == None:
        args.checkpoint = -1
    load_run = export_policy(args)
    export_config(load_run)
    