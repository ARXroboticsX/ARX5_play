# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import argparse
import os
import sys
import threading
import yaml

import torch

from utils.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils.utils import set_seed  # helper functions

sys.path.append("./")

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def get_model_config(opt):
    config = {
        'ckpt_dir': opt.ckpt_dir,
        'ckpt_name': opt.ckpt_name,
        'onnx_dir': opt.onnx_dir,
        'onnx_name': opt.onnx_name,
        'use_depth_image': opt.use_depth_image,
        'use_robot_base': opt.use_robot_base
    }

    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def export_model(config):
    # 创建模型数据
    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)

    # 加载模型权重
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}

    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value

    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    # 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    camera_num = len(config['camera_names'])

    image = torch.randn(1, camera_num, 3, 480, 640).cuda()
    if config['policy_config'].get('use_depth_image', True):
        depth_image = torch.randn(1, camera_num, 3, 480, 640).cuda()
    else:
        depth_image = None
    if config['policy_config'].get('use_robot_base', True):
        state = torch.randn(1, 16).cuda()
    else:
        state = torch.randn(1, 14).cuda()

    # 导出onnxx
    if not os.path.exists(config['onnx_dir']):
        os.makedirs(config['onnx_dir'])
    onnx_path = os.path.join(config['onnx_dir'], config['onnx_name'])
    torch.onnx.export(policy, (image, depth_image, state), onnx_path)
    print('Export to ONNX: {}'.format(onnx_path))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir', type=str, help='ckpt dir', default='./weights')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt', help='ckpt name')

    parser.add_argument('--onnx_dir', type=str, help='onnx dir', default='./weights')
    parser.add_argument('--onnx_name', type=str, default='policy_best.onnx', help='onnx name')

    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'],
                        default='ACT', help='policy class, capitalize, CNNMLP, ACT, Diffusion')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], help='camera names')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--use_robot_base', action='store_true', help='use robot base')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main():
    opt = parse_opt()
    config = get_model_config(opt)
    export_model(config)


if __name__ == '__main__':
    main()
