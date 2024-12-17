from humanoid import LEGGED_GYM_ROOT_DIR
import os
from humanoid.envs import *
from humanoid.utils import  get_args, task_registry
from datetime import datetime
import torch
from export_mnn_t1 import export_mnn_t1

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

    models = [file for file in os.listdir(load_run)]
    models.sort(key=lambda m: "{0:0>15}".format(m))
    model = models[-1]
    # 这行不加上路径不对
    load_path = os.path.join(load_run, model,'policy_dh.jit')
    # print(load_path)
    return load_path

def export_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # load jit
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,'')
    # print(log_root)
    model_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    # print(model_path)
    # print("Load model from:", model_path)
    jit_model = torch.jit.load(model_path)
    jit_model.eval()

    current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    onnx_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_onnx',
                            current_date_time)
    os.makedirs(onnx_dir, exist_ok=True)
    # dir_name = args.task.split('_')[0] + "_policy.onnx"
    onnx_file_name = "ti5_dh_policy.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_file_name)
    # 创建模型的示例输入（根据模型需求调整大小和类型）
    # example_input = torch.randn(1,302)  # 假设这是适合第一个模型的输入
    # example_input = torch.randn(1,env_cfg.env.num_single_obs*env_cfg.env.short_frame_stack+64+3)  # 假设这是适合第一个模型的输入
    example_input = torch.randn(1,env_cfg.env.num_observations)  # 假设这是适合第一个模型的输入
    # 导出模型
    torch.onnx.export(jit_model,               # JIT 模型
                    example_input,             # 模型的示例输入
                    onnx_path,                 # 模型的输出路径
                    export_params=True,        # 导出模型的参数
                    opset_version=11,          # ONNX opset 版本
                    do_constant_folding=True,  # 优化常量折叠
                    input_names=['input'],     # 输入名
                    output_names=['output'],   # 输出名
                    # output_names=['output','est_vel'],   # 输出名
                    )
    print("Export onnx model to: ", onnx_path)

    # export mnn model
    mnn_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_mnn')
    export_mnn_t1(mnn_dir, onnx_path)

if __name__ == '__main__':
    args = get_args()
    if args.load_run == None:
        args.load_run = -1
    export_onnx(args)