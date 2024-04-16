# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env
import collections
from collections import deque
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2
import yaml


# 读取相关配置文件
def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def compress_images(image_list, encode_param):
    compressed_list = []
    compressed_len = []

    for image in image_list:
        result, encoded_image = cv2.imencode('.jpg', image, encode_param)
        compressed_list.append(encoded_image)
        compressed_len.append(len(encoded_image))

    return compressed_list, compressed_len


def pad_images(compressed_image_list, padded_size):
    padded_compressed_image_list = []

    for compressed_image in compressed_image_list:
        padded_compressed_image = np.zeros(padded_size, dtype='uint8')
        image_len = len(compressed_image)
        padded_compressed_image[:image_len] = compressed_image
        padded_compressed_image_list.append(padded_compressed_image)

    return padded_compressed_image_list


# 保存数据函数
def save_data(opt, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        # '/base_action_t265': [],
    }

    # 相机字典  观察的图像
    for cam_name in opt.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if opt.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)  # 动作  当前动作
        ts = timesteps.pop(0)  # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in opt.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(
                ts.observation['images'][cam_name])
            if opt.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(
                    ts.observation['images_depth'][cam_name])

    # 压缩图像
    if opt.is_compress:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 压缩质量
        compressed_len = []

        for cam_name in opt.camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])  # 压缩的长度

            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param)

                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))

            # 更新图像
            data_dict[f'/observations/images/{cam_name}'] = compressed_list

        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()  # 取最大的图像长度，图像压缩后就是一个buf序列

        for cam_name in opt.camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)

            # 更新压缩后的图像列表
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

        if opt.use_depth_image:
            compressed_len_depth = []

            for cam_name in opt.camera_names:
                depth_list = data_dict[f'/observations/depths/{cam_name}']
                compressed_list_depth = []
                compressed_len_depth.append([])  # 压缩的长度

                for depth in depth_list:
                    result, encoded_depth = cv2.imencode('.jpg', depth, encode_param)

                    compressed_list_depth.append(encoded_depth)
                    compressed_len_depth[-1].append(len(encoded_depth))

                # 更新图像
                data_dict[f'/observations/depths/{cam_name}'] = compressed_list_depth

            compressed_len_depth = np.array(compressed_len_depth)
            padded_size_depth = compressed_len_depth.max()

            for cam_name in opt.camera_names:
                compressed_depth_list = data_dict[f'/observations/depths/{cam_name}']
                padded_compressed_depth_list = []
                for compressed_depth in compressed_depth_list:
                    padded_compressed_depth = np.zeros(padded_size_depth, dtype='uint8')
                    depth_len = len(compressed_depth)
                    padded_compressed_depth[:depth_len] = compressed_depth
                    padded_compressed_depth_list.append(padded_compressed_depth)
                data_dict[f'/observations/depths/{cam_name}'] = padded_compressed_depth_list

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩

        root.attrs['sim'] = False
        root.attrs['compress'] = False
        if opt.is_compress:
            root.attrs['compress'] = True

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        depth = obs.create_group('depths')

        for cam_name in opt.camera_names:
            if opt.is_compress:
                image_shape = (opt.max_timesteps, padded_size)
                image_chunks = (1, padded_size)

                if opt.use_depth_image:
                    depth_shape = (opt.max_timesteps, padded_size_depth)
                    depth_chunks = (1, padded_size_depth)
            else:
                image_shape = (opt.max_timesteps, 480, 640, 3)
                image_chunks = (1, 480, 640, 3)

                if opt.use_depth_image:
                    depth_shape = (opt.max_timesteps, 480, 640)
                    depth_chunks = (1, 480, 640)

            _ = image.create_dataset(cam_name, image_shape, 'uint8', chunks=image_chunks)
            if opt.use_depth_image:
                _ = depth.create_dataset(cam_name, depth_shape, 'uint16', chunks=depth_chunks)

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        # data_dict写入h5py.File
        for name, array in data_dict.items():  # 名字+值
            root[name][...] = array

    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)


class RosOperator:
    def __init__(self, opt, config):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.opt = opt
        self.config = config
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

    def get_frame(self):
        if (len(self.img_left_deque) == 0 or
                len(self.img_right_deque) == 0 or
                len(self.img_front_deque) == 0 or
                (self.opt.use_depth_image and (
                        len(self.img_left_depth_deque) == 0 or
                        len(self.img_right_depth_deque) == 0 or
                        len(self.img_front_depth_deque) == 0))):
            return False
        if self.opt.use_depth_image:
            frame_time = min(
                [self.img_left_deque[-1].header.stamp.to_sec(),
                 self.img_right_deque[-1].header.stamp.to_sec(),
                 self.img_front_deque[-1].header.stamp.to_sec(),
                 self.img_left_depth_deque[-1].header.stamp.to_sec(),
                 self.img_right_depth_deque[-1].header.stamp.to_sec(),
                 self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min(
                [self.img_left_deque[-1].header.stamp.to_sec(),
                 self.img_right_deque[-1].header.stamp.to_sec(),
                 self.img_front_deque[-1].header.stamp.to_sec()])

        if (len(self.img_left_deque) == 0 or
                self.img_left_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.img_right_deque) == 0 or
                self.img_right_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.img_front_deque) == 0 or
                self.img_front_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.master_arm_left_deque) == 0 or
                self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.master_arm_right_deque) == 0 or
                self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.puppet_arm_left_deque) == 0 or
                self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (len(self.puppet_arm_right_deque) == 0 or
                self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if (self.opt.use_depth_image and
                (len(self.img_left_depth_deque) == 0 or
                 self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time)):
            return False
        if (self.opt.use_depth_image and
                (len(self.img_right_depth_deque) == 0 or
                 self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time)):
            return False
        if (self.opt.use_depth_image and
                (len(self.img_front_depth_deque) == 0 or
                 self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time)):
            return False
        if (self.opt.use_robot_base and
                (len(self.robot_base_deque) == 0 or
                 self.robot_base_deque[-1].header.stamp.to_sec() < frame_time)):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        if self.opt.is_compress:
            img_left = self.bridge.compressed_imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        else:
            img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        if self.opt.is_compress:
            img_right = self.bridge.compressed_imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')
        else:
            img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        if self.opt.is_compress:
            img_front = self.bridge.compressed_imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')
        else:
            img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        img_left_depth = []
        if self.opt.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            if self.opt.is_compress:
                img_left_depth = self.bridge.compressed_imgmsg_to_cv2(self.img_left_depth_deque.popleft(),
                                                                      'passthrough')
            else:
                img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
        # top, bottom, left, right = 40, 40, 0, 0
        # img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom,
        #                                     left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = []
        if self.opt.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            if self.opt.is_compress:
                img_right_depth = self.bridge.compressed_imgmsg_to_cv2(self.img_right_depth_deque.popleft(),
                                                                       'passthrough')
            else:
                img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
        # top, bottom, left, right = 40, 40, 0, 0
        # img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom,
        #                                      left, right, cv2.BORDER_CONSTANT, value=0)

        img_front_depth = []
        if self.opt.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            if self.opt.is_compress:
                img_front_depth = self.bridge.compressed_imgmsg_to_cv2(self.img_front_depth_deque.popleft(),
                                                                       'passthrough')
            else:
                img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')
        # top, bottom, left, right = 40, 40, 0, 0
        # img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom,
        #                                      left, right, cv2.BORDER_CONSTANT, value=0)

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        robot_base = None
        if self.opt.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)

        image_type = 'compress_image' if self.opt.is_compress else 'original_image'
        callback_type = CompressedImage if self.opt.is_compress else Image

        rospy.Subscriber(self.config['camera_config'][image_type]['img_left_topic'],
                         callback_type, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['camera_config'][image_type]['img_right_topic'],
                         callback_type, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['camera_config'][image_type]['img_front_topic'],
                         callback_type, self.img_front_callback, queue_size=1000, tcp_nodelay=True)

        if self.opt.use_depth_image:
            rospy.Subscriber(self.config['camera_config'][image_type]['img_left_depth_topic'],
                             callback_type, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.config['camera_config'][image_type]['img_right_depth_topic'],
                             callback_type, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.config['camera_config'][image_type]['img_front_depth_topic'],
                             callback_type, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.config['arm_config']['master_arm_left_topic'],
                         JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['master_arm_right_topic'],
                         JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['puppet_arm_left_topic'],
                         JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['puppet_arm_right_topic'],
                         JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.config['base_config']['robot_base_topic'],
                         Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):
        timesteps = []
        actions = []
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.opt.camera_names:
            image_dict[cam_name] = image
        count = 0

        # input_key = input("please input s:")
        # while input_key != 's' and not rospy.is_shutdown():
        #     input_key = input("please input s:")

        rate = rospy.Rate(self.opt.frame_rate)
        print_flag = True

        global exit_flag
        while (count < self.opt.max_timesteps + 1) and not rospy.is_shutdown():
            # 2 收集数据
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue

            print_flag = True
            count += 1
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result

            # 2.1 图像信息
            camera_map = {
                'cam_high': img_front,
                'cam_left_wrist': img_left,
                'cam_right_wrist': img_right,
            }

            image_dict = dict()
            for camera_name, image in camera_map.items():
                if camera_name in self.opt.camera_names:
                    image_dict[camera_name] = image

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.opt.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.opt.camera_names[0]] = img_front_depth
                image_dict_depth[self.opt.camera_names[1]] = img_left_depth
                image_dict_depth[self.opt.camera_names[2]] = img_right_depth
                obs['images_depth'] = image_dict_depth

            obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position),
                                          np.array(puppet_arm_right.position)),
                                         axis=0)
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity),
                                          np.array(puppet_arm_right.velocity)),
                                         axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort),
                                            np.array(puppet_arm_right.effort)),
                                           axis=0)

            if self.opt.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            action = np.concatenate((np.array(master_arm_left.position),
                                     np.array(master_arm_right.position)), axis=0)
            actions.append(action)
            timesteps.append(ts)
            print("Frame data: ", count)
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))

        return timesteps, actions


def main(opt):
    config = load_yaml(opt.data)
    ros_operator = RosOperator(opt, config)
    timesteps, actions = ros_operator.process()

    if(len(actions) < opt.max_timesteps):
        print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %opt.max_timesteps)
        exit(-1)

    if not os.path.exists(opt.datasets):
        os.makedirs(opt.datasets)
    dataset_path = os.path.join(opt.datasets, "episode_" + str(opt.episode_idx))
    save_data(opt, timesteps, actions, dataset_path)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str, default="./datasets", help='dataset dir')
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')
    parser.add_argument('--max_timesteps', type=int, default=600, help='max timesteps')
    parser.add_argument('--frame_rate', type=int, default=90, help='frame rate')

    parser.add_argument('--data', type=str, default="./data/config.yaml", help='config file')

    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], help='camera names')

    parser.add_argument('--use_robot_base', action='store_true', help='use robot base')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')

    # 是否压缩图像
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
