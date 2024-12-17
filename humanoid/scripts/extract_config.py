import ast
import yaml
from collections import OrderedDict

# 键名映射
KEY_MAPPING = {
    '1_joint': ['leg_l1_joint', 'leg_r1_joint'],
    '2_joint': ['leg_l2_joint', 'leg_r2_joint'],
    '3_joint': ['leg_l3_joint', 'leg_r3_joint'],
    '4_joint': ['leg_l4_joint', 'leg_r4_joint'],
    '5_joint': ['leg_l5_joint', 'leg_r5_joint'],
    '6_joint': ['leg_l6_joint', 'leg_r6_joint'],
}

class ConfigExtractor(ast.NodeVisitor):
    def __init__(self, context=None):
        if context is None:
            context = {}
        self.context = context
        self.parameters = OrderedDict({
            'LeggedRobotCfg': OrderedDict({
                'init_state': OrderedDict({
                    'default_joint_angle': {}
                }),
                'control': OrderedDict({
                    'stiffness': {},
                    'damping': {},
                    'action_scale': None,
                    'decimation': None,
                    'cycle_time': None
                }),
                'normalization': OrderedDict({
                    'clip_scales': OrderedDict({
                        'clip_observations': None,
                        'clip_actions': None
                    }),
                    'obs_scales': {}
                }),
                'size': OrderedDict({
                    'actions_size': None,
                    'observations_size': None,
                    'num_hist': 66,
                }),
                'mode': OrderedDict({
                    'sw_mode': True,
                    'cmd_threshold': 0.05,
                    'ang_vel_threshold': 100000,
                    'angle_threshold': 0.1
                }),
                'filter': OrderedDict({
                    'filt_action': True,
                    'sample_rate': 100,
                    'cutoff_freq': 3.0
                })
            })
        })

    def visit_ClassDef(self, node):
        if node.name == 'init_state':
            self.extract_init_state(node)
        elif node.name == 'control':
            self.extract_control(node)
        elif node.name == 'normalization':
            self.extract_normalization(node)
        elif node.name == 'DHT1StandCfg':
            self.extract_size(node)
            self.extract_cycle_time(node)
            
        self.generic_visit(node)

    def extract_init_state(self, node):
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == 'default_joint_angles':
                        self.parameters['LeggedRobotCfg']['init_state']['default_joint_angle'] = self.safe_eval(child.value)

    def extract_control(self, node):
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'stiffness':
                            self.parameters['LeggedRobotCfg']['control']['stiffness'] = self.map_keys(self.safe_eval(child.value), KEY_MAPPING)
                        elif target.id == 'damping':
                            self.parameters['LeggedRobotCfg']['control']['damping'] = self.map_keys(self.safe_eval(child.value), KEY_MAPPING)
                        elif target.id == 'action_scale':
                            self.parameters['LeggedRobotCfg']['control']['action_scale'] = self.safe_eval(child.value)
                        elif target.id == 'decimation':
                            self.parameters['LeggedRobotCfg']['control']['decimation'] = self.safe_eval(child.value)
                        elif target.id == 'cycle_time':
                            self.parameters['LeggedRobotCfg']['control']['cycle_time'] = self.safe_eval(child.value)

    def extract_normalization(self, node):
        for child in node.body:
            if isinstance(child, ast.ClassDef) and child.name == 'obs_scales':
                for grandchild in child.body:
                    if isinstance(grandchild, ast.Assign):
                        for target in grandchild.targets:
                            self.parameters['LeggedRobotCfg']['normalization']['obs_scales'][target.id] = self.safe_eval(grandchild.value)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if target.id == 'clip_observations':
                        self.parameters['LeggedRobotCfg']['normalization']['clip_scales']['clip_observations'] = self.safe_eval(child.value)
                    elif target.id == 'clip_actions':
                        self.parameters['LeggedRobotCfg']['normalization']['clip_scales']['clip_actions'] = self.safe_eval(child.value)

    def extract_size(self, node):
        for child in node.body:
            if isinstance(child, ast.ClassDef) and child.name == 'env':
                for grandchild in child.body:
                    if isinstance(grandchild, ast.Assign):
                        for target in grandchild.targets:
                            if target.id == 'num_actions':
                                self.parameters['LeggedRobotCfg']['size']['actions_size'] = self.safe_eval(grandchild.value)
                            elif target.id == 'num_single_obs':
                                self.parameters['LeggedRobotCfg']['size']['observations_size'] = self.safe_eval(grandchild.value)
    def extract_cycle_time(self, node):
        for child in node.body:
            if isinstance(child, ast.ClassDef) and child.name == 'rewards':
                for grandchild in child.body:
                    if isinstance(grandchild, ast.Assign):
                        for target in grandchild.targets:
                            if target.id == 'cycle_time':
                                self.parameters['LeggedRobotCfg']['control']['cycle_time'] = self.safe_eval(grandchild.value)

    def safe_eval(self, node):
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.List):
                return [self.safe_eval(el) for el in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(self.safe_eval(el) for el in node.elts)
            elif isinstance(node, ast.Set):
                return {self.safe_eval(el) for el in node.elts}
            elif isinstance(node, ast.Dict):
                return {self.safe_eval(key): self.safe_eval(value) for key, value in zip(node.keys, node.values)}
            elif isinstance(node, ast.BinOp) or isinstance(node, ast.UnaryOp):
                return eval(compile(ast.Expression(node), filename="<ast>", mode="eval"), {}, self.context)
            elif isinstance(node, ast.Name):
                return self.context.get(node.id, None)
            else:
                return ast.literal_eval(node)
        except Exception as e:
            print(f"Error evaluating node {ast.dump(node)}: {e}")
            return None

    def map_keys(self, data, key_mapping):
        if isinstance(data, dict):
            new_data = OrderedDict()
            for k, v in data.items():
                if k in key_mapping:
                    mapped_keys = key_mapping[k]
                    if isinstance(mapped_keys, list):
                        for mapped_key in mapped_keys:
                            new_data[mapped_key] = self.map_keys(v, key_mapping)
                    else:
                        new_data[mapped_keys] = self.map_keys(v, key_mapping)
                else:
                    new_data[k] = self.map_keys(v, key_mapping)
            return new_data
        elif isinstance(data, list):
            return [self.map_keys(item, key_mapping) for item in data]
        else:
            return data

def extract_context_from_file(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    
    context = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'init_angle':
                    context['init_angle'] = node.value.value  # 假设init_angle是一个常量，如果是表达式，请使用safe_eval方法

    return context

def extract_parameters_from_file(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    context = extract_context_from_file(file_path)
    extractor = ConfigExtractor(context)
    extractor.visit(tree)
    return extractor.parameters

def save_parameters_to_yaml(parameters, output_file):
    if not output_file.endswith('.yaml'):
        output_file += '.yaml'

    class OrderedDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(OrderedDumper, self).increase_indent(flow, False)

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    OrderedDumper.add_representer(OrderedDict, dict_representer)

    # with open(output_file, 'w') as yaml_file:
    #     yaml.dump(parameters, yaml_file, Dumper=OrderedDumper, default_flow_style=False)

# 示例用法
# parameters = extract_parameters_from_file('input_file.py')
# save_parameters_to_yaml(parameters, 'output_file.yaml')


