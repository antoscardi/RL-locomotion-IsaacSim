#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import torch

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Twist



class WalkRLNetwork(Node):
  def __init__(self):
    super().__init__('walk_rk_network')

    # Storage for the latest values
    self.vx_ = self.vy_ = self.vz_ = 0.0
    self.wx_ = self.wy_ = self.wz_ = 0.0
    # store velocities as vectors
    self.linear_vel_ = np.zeros(3)   # [vx, vy, vz]
    self.angular_vel_ = np.zeros(3)  # [wx, wy, wz]
    self.joint_pos_ = np.zeros(43, dtype=np.float32)
    self.joint_vel_ = np.zeros(43, dtype=np.float32)
    self.commands_ = np.zeros(43, dtype=np.float32)
    # last output of the network
    # self.last_action_ = np.zeros(37, dtype=np.float32)
    self.height_scan_big_ = np.array([])    # do the init allocation (160 already? with dim check in the function?)
    self.height_scan_small_ = np.array([])
    
    # input tensor for the NN
    # self.input_obs_tensor_ = torch.zeros(310, dtype=torch.float32)

    # Subscribers
    self.create_subscription(Odometry, '/odom_robot', self.odom_callback, 10)
    self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
    self.create_subscription(Twist, '/cmd_vel', self.joint_state_callback, 10)
    self.create_subscription(GridMap, '/elevation_map', self.elevation_map_callback, 1)
    self.create_subscription(GridMap, '/elevation_map_small', self.elevation_map_small_callback, 1)

    self.create_timer(0.1, self.timer_callback)


  def odom_callback(self, msg: Odometry):
    self.vx_, self.vy_, self.vz_ = (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z)
    self.wx_, self.wy_, self.wz_ = (msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z)
    # store into numpy vectors
    self.linear_vel_ = np.array((self.vx_, self.vy_, self.vz_))
    self.angular_vel_ = np.array((self.wx_, self.wy_, self.wz_))

  def joint_state_callback(self, msg: JointState):
    if len(msg.position) != 43:
      self.get_logger().warn(f"JointState size mismatch: expected 44, got {len(msg.position)}")
      return
    self.joint_pos_[:] = msg.position
    self.joint_vel_[:] = msg.velocity
  
  # # To understand the topic of Twist msg
  # def commands_callback(self, msg: Twist):
  #   self.commands_ = 
  
  # # In theory not necessary because there is already the class variable assigned in the network function (its output)
  # def last_action(self):
  #   self.last_action_ = ...

  def elevation_map_callback(self, msg: GridMap):
    self.height_scan_big_ = self.flatten_layer(msg, 'elevation')

  def elevation_map_small_callback(self, msg: GridMap):
    self.height_scan_small_ = self.flatten_layer(msg, 'elevation')

  
  def flatten_layer(self, msg: GridMap, layer_name: str) -> np.ndarray:
    try:
      idx = msg.layers.index(layer_name)
    except ValueError:
      self.get_logger().warn(f"Layer '{layer_name}' not in {msg.layers}")
      return np.array([])

    # msg.data[idx] is a MultiArray-like container for that one layer:
    layer_data_array = msg.data[idx]
    # layer_data_array is the flat list of floats (length cells_per_layer)
    values = np.array(layer_data_array.data, dtype=float)
    values = np.nan_to_num(values, nan=-0.8)
    # to drop all the NaN values
    # values = values[~np.isnan(values)]
    
    return values
  
  def print_row(self, name, vec):
    print(f"{name} [{len(vec)}]: \n", *(f"{x:.3f}" for x in vec))
    
  def build_observation(self) -> torch.Tensor:

    obs_vec = np.concatenate((
      self.linear_vel_,
      self.angular_vel_,
      self.joint_pos_,
      self.joint_vel_,
      self.height_scan_big_,
      self.height_scan_small_,
    ), axis=0)

    # if obs_vec.shape[0] != self.input_obs_tensor_.shape[0]:
    #   self.get_logger().warn(f"Obs shape mismatch: got {obs_vec.shape[0]}, expected {self.input_obs_tensor_.shape[0]}")

    return torch.from_numpy(obs_vec).float()

  
  def timer_callback(self):
    # build the tensor of the input of the nn, dim: (310,)
    # self.input_obs_tensor_[:] = self.build_observation()
    self.input_obs_tensor_ = self.build_observation()
    self.print_row(">>> TENSOR",   self.input_obs_tensor_)
    
    self.get_logger().info(
      f"Obs: \nlinear_vel=({self.vx_:.2f},{self.vy_:.2f},{self.vz_:.2f}), \n"
      f"angular_vel=({self.wx_:.2f},{self.wy_:.2f},{self.wz_:.2f}), \n"
      f"jpos={len(self.joint_pos_)}, \n"
      f"h_big={self.height_scan_big_.shape}, \nh_small={self.height_scan_small_.shape}\n"
      f"full_obs_vector={self.input_obs_tensor_.shape}  \n"
    )
    self.print_row(">>> big_elevation_map",   self.height_scan_big_)
    self.print_row(">>> small_elevation_map", self.height_scan_small_)
    print("")
  


def main(args=None):
  rclpy.init(args=args)
  node = WalkRLNetwork()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

