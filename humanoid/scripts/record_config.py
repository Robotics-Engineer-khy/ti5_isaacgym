import os
from humanoid import LEGGED_GYM_ENVS_DIR,LEGGED_GYM_ROOT_DIR

# 记录与双足机器人模拟相关的配置文件
def record_config(log_root, urdf_path, name="a1_amp"):
    # 日志的根路径，函数将把配置文件保存到该目录
    log_dir=log_root
    # 创建日志目录，如果目录已经存在，则不会报错
    os.makedirs(log_dir, exist_ok=True)

    # 对应't1_dh_stand_config.txt'
    str_config = name + '_config.txt'
    file_path1=os.path.join(log_dir, str_config)

    # 对应'legged_robot_config.txt'文件
    file_path2=os.path.join(log_dir, 'legged_robot_config.txt')
    # 对应'legged_robot.txt'文件
    file_path5=os.path.join(log_dir, 'legged_robot.txt')

    # 提取urdf文件名为't1.urdf'
    str_urdf = urdf_path.split('/')[-1]
    # 对应't1_urdf'文件
    file_path3=os.path.join(log_dir, str_urdf)

    # 对应't1_dh_stand_env.txt'文件
    str_config1 = name + '_env.txt'
    file_path4=os.path.join(log_dir, str_config1)

    # 取't1'
    root1 = name.split('_')[0][:2]

    # 路径为'humanoid/envs/t1/t1_config.py'
    root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '_config.py')
    # 路径为'humanoid/envs/base/legged_robot_config.py'
    root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot_config.py')
    # 路径为'resources/robots/t1/urdf/t1.urdf'
    root_path3 = urdf_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    # 路径为'humanoid/envs/t1/t1_env.py'
    root_path4 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '_env.py')
    # 路径为'humanoid/envs/base/legged_robot.py'
    root_path5 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot.py')
    

    # with open(root_path1, 'r', encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path1, 'w', encoding='utf-8') as file:
    #     file.write(content)

    # with open(root_path2, 'r',encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path2, 'w', encoding='utf-8') as file:
    #     file.write(content)
        
    # with open(root_path3, 'r',encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path3, 'w', encoding='utf-8') as file:
    #     file.write(content)
    
    # if os.path.exists(root_path4):
    #     with open(root_path4, 'r',encoding='utf-8') as file:
    #         content = file.read()

    #     with open(file_path4, 'w', encoding='utf-8') as file:
    #         file.write(content)
            
    # with open(root_path5, 'r',encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path5, 'w', encoding='utf-8') as file:
    #     file.write(content)

    # 将源文件路径和目标文件路径放入列表中
    root_paths = [root_path1, root_path2, root_path3, root_path4, root_path5]
    file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5]

    # 使用zip将源路径和目标路径配对进行迭代
    for root_path, file_path in zip(root_paths, file_paths):
        # 对于每一个源路径，首先检查文件是否存在
        if os.path.exists(root_path):
            # 如果存在，则打开文件读取其内容
            with open(root_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # 然后在目标路径中创建文件，并写入读取的内容
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)