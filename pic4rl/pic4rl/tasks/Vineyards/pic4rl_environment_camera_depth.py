#!/usr/bin/env python3

import os
import numpy as np
from numpy import savetxt
import math
import subprocess
import json
import random
import sys
import time
import datetime
import yaml
import logging
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors
from pic4rl.utils.env_utils import *
from pic4rl.testing.nav_metrics import Navigation_Metrics


class Pic4rlEnvironmentCamera(Node):
    def __init__(self):
        """ """
        super().__init__("pic4rl_training_vineyard_camera")
        self.declare_parameter("package_name", "pic4rl")
        self.declare_parameter("training_params_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("main_params_path", rclpy.Parameter.Type.STRING)
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        goals_path = os.path.join(
            get_package_share_directory(self.package_name), "goals_and_poses"
        )
        self.main_params_path = self.get_parameter("main_params_path").get_parameter_value().string_value
        train_params_path = self.get_parameter("training_params_path").get_parameter_value().string_value
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), "models/goal_box/model.sdf"
        )

        with open(train_params_path, "r") as train_param_file:
            train_params = yaml.safe_load(train_param_file)["training_params"]

        self.declare_parameters(
            namespace="",
            parameters=[
                ("mode", rclpy.Parameter.Type.STRING),
                ("data_path", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("visual_data", rclpy.Parameter.Type.STRING),
                ("features", rclpy.Parameter.Type.INTEGER),
                ("channels", rclpy.Parameter.Type.INTEGER),
                ("depth_param.width", rclpy.Parameter.Type.INTEGER),
                ("depth_param.height", rclpy.Parameter.Type.INTEGER),
                ("depth_param.dist_cutoff", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
                ("update_frequency", rclpy.Parameter.Type.DOUBLE),
                ("sensor", rclpy.Parameter.Type.STRING),
            ],
        )

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        goals_path = os.path.join(goals_path, self.mode)
        self.data_path = (
            self.get_parameter("data_path").get_parameter_value().string_value
        )
        self.data_path = os.path.join(goals_path, self.data_path)
        print(train_params["--change_goal_and_pose"])
        self.change_episode = int(train_params["--change_goal_and_pose"])
        self.starting_episodes = int(train_params["--starting_episodes"])
        self.timeout_steps = int(train_params["--episode-max-steps"])
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        self.goal_tolerance = (
            self.get_parameter("goal_tolerance").get_parameter_value().double_value
        )
        self.visual_data = (
            self.get_parameter("visual_data").get_parameter_value().string_value
        )
        self.features = (
            self.get_parameter("features").get_parameter_value().integer_value
        )
        self.channels = (
            self.get_parameter("channels").get_parameter_value().integer_value
        )
        self.image_width = (
            self.get_parameter("depth_param.width").get_parameter_value().integer_value
        )
        self.image_height = (
            self.get_parameter("depth_param.height").get_parameter_value().integer_value
        )
        self.max_depth = (
            self.get_parameter("depth_param.dist_cutoff").get_parameter_value().double_value
        )
        self.lidar_distance = (
            self.get_parameter("laser_param.max_distance")
            .get_parameter_value()
            .double_value
        )
        self.lidar_points = (
            self.get_parameter("laser_param.num_points")
            .get_parameter_value()
            .integer_value
        )
        self.update_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
        )
        self.sensor_type = (
            self.get_parameter("sensor").get_parameter_value().string_value
        )

        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        log_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--logdir"])

        self.logdir = create_logdir(
            train_params["--policy"], self.sensor_type, log_path
        )
        self.get_logger().info(f"Logdir: {self.logdir}")

        if "--model-dir" in train_params:
            self.model_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--model-dir"])
        self.spin_sensors_callbacks()

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", qos)

        self.episode_step = 0
        self.previous_twist = Twist()
        self.episode = 0
        self.collision_count = 0
        self.t0 = 0.0
        self.evaluate = True
        self.index = 0

        self.get_logger().info(f"Gym mode: {self.mode}")
        self.get_logger().debug("PIC4RL_Environment: Starting process")

    def step(self, action, episode_step=0):
        """ """
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.episode_step = episode_step

        observation, reward, done = self._step(twist)
        info = None

        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step=False):
        """ """
        self.get_logger().debug("sending action...")
        self.send_action(twist)

        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
        
        depth_image = self.get_sensor_data()

        if not reset_step:
            self.get_logger().debug("checking events...")
            done, event = self.check_events()

            reward = 0.

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(
                twist, depth_image
            )
        else:
            reward = None
            observation = None
            done = False
            event = None

        # Send observation and reward
        self.update_state(twist, depth_image, done, event)
        if done:
            time.sleep(1.5)

        return observation, reward, done

    def get_goals_and_poses(self):
        """ """
        data = json.load(open(self.data_path, "r"))

        return data["initial_pose"], data["goals"], data["poses"]

    def spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            empty_measurements = [ k for k, v in self.sensors.sensor_msg.items() if v is None]
            self.get_logger().debug(f"empty_measurements: {empty_measurements}")
            rclpy.spin_once(self)
            self.get_logger().debug("spin once ...")
        self.get_logger().debug("spin sensor callback complete ...")
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def send_action(self, twist):
        """ """
        self.cmd_vel_pub.publish(twist)
        # Regulate frequency of send action if needed
        freq, t1 = compute_frequency(self.t0)
        self.get_logger().debug(f"frequency : {freq}")
        self.t0 = t1
        if freq > self.update_freq:
            frequency_control(self.update_freq)

        # self.get_logger().debug("pausing...")
        # self.pause()

    def get_sensor_data(self):
        """ """
        sensor_data = {}
        sensor_data["depth"] = self.sensors.get_depth()

        if sensor_data["depth"] is None:
            sensor_data["depth"] = (
                np.ones((self.image_height, self.image_width, 1)) * self.max_depth
            )

        self.get_logger().debug("processing odom...")
        depth_image = sensor_data["depth"]

        return depth_image

    def check_events(self,):

        if self.episode_step + 1 >= self.timeout_steps:
            self.get_logger().info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            logging.info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            return True, "timeout"

        return False, "None"


    def get_observation(self, twist, depth_image):
        """ """
        # flattened depth image
        if self.visual_data == "features":
            features = depth_image.flatten()

        # previous velocity state
        v = twist.linear.x
        w = twist.angular.z
        vel = np.array([v, w], dtype=np.float32)
        state = np.concatenate((vel, features))
        return state

    def update_state(self, twist, depth_image, done, event):
        """ """
        self.previous_twist = twist
        self.previous_depth_image = depth_image

    def reset(self, n_episode, tot_steps, evaluate=False):
        """ """
        self.episode = n_episode
        self.evaluate = evaluate
        logging.info(
            f"Total_episodes: {n_episode}{' evaluation episode' if self.evaluate else ''}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n"
        )
        print()
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")

        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

        ( _, _, _) = self._step(reset_step=True)
        (observation, _, _) = self._step()

        return observation