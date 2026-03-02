#!/usr/bin/env python3
"""ROS2 node for walking RL network observation builder.

Subscribes to odometry, joint states, velocity commands, and elevation
maps, then assembles the full observation vector for the policy network.
"""

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


class WalkRLNetwork(Node):
    """Collects sensor data and builds the observation tensor."""

    def __init__(self):
        super().__init__('walk_rk_network')

        # Storage for the latest values
        self.vx_ = self.vy_ = self.vz_ = 0.0
        self.wx_ = self.wy_ = self.wz_ = 0.0

        # Store velocities as vectors
        self.linear_vel_ = np.zeros(3)
        self.angular_vel_ = np.zeros(3)
        self.joint_pos_ = np.zeros(43, dtype=np.float32)
        self.joint_vel_ = np.zeros(43, dtype=np.float32)
        self.commands_ = np.zeros(43, dtype=np.float32)
        self.height_scan_big_ = np.array([])
        self.height_scan_small_ = np.array([])

        # Subscribers
        self.create_subscription(
            Odometry, '/odom_robot', self.odom_callback, 10,
        )
        self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10,
        )
        self.create_subscription(
            Twist, '/cmd_vel', self.joint_state_callback, 10,
        )
        self.create_subscription(
            GridMap, '/elevation_map',
            self.elevation_map_callback, 1,
        )
        self.create_subscription(
            GridMap, '/elevation_map_small',
            self.elevation_map_small_callback, 1,
        )

        self.create_timer(0.1, self.timer_callback)

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        """Store linear and angular velocity from odometry."""
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular

        self.vx_ = linear.x
        self.vy_ = linear.y
        self.vz_ = linear.z
        self.wx_ = angular.x
        self.wy_ = angular.y
        self.wz_ = angular.z

        self.linear_vel_ = np.array(
            (self.vx_, self.vy_, self.vz_),
        )
        self.angular_vel_ = np.array(
            (self.wx_, self.wy_, self.wz_),
        )

    def joint_state_callback(self, msg: JointState):
        """Store joint positions and velocities."""
        if len(msg.position) != 43:
            self.get_logger().warn(
                f'JointState size mismatch: expected 43, '
                f'got {len(msg.position)}',
            )
            return
        self.joint_pos_[:] = msg.position
        self.joint_vel_[:] = msg.velocity

    def elevation_map_callback(self, msg: GridMap):
        """Flatten the large elevation map layer."""
        self.height_scan_big_ = self._flatten_layer(
            msg, 'elevation',
        )

    def elevation_map_small_callback(self, msg: GridMap):
        """Flatten the small elevation map layer."""
        self.height_scan_small_ = self._flatten_layer(
            msg, 'elevation',
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_layer(
        self,
        msg: GridMap,
        layer_name: str,
    ) -> np.ndarray:
        """Extract and flatten a named layer from a GridMap message."""
        try:
            idx = msg.layers.index(layer_name)
        except ValueError:
            self.get_logger().warn(
                f"Layer '{layer_name}' not in {msg.layers}",
            )
            return np.array([])

        layer_data_array = msg.data[idx]
        values = np.array(layer_data_array.data, dtype=float)
        values = np.nan_to_num(values, nan=-0.8)
        return values

    @staticmethod
    def _print_row(name: str, vec) -> None:
        """Pretty-print a labelled vector."""
        formatted = ' '.join(f'{x:.3f}' for x in vec)
        print(f'{name} [{len(vec)}]: {formatted}')

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(self) -> torch.Tensor:
        """Concatenate all sensor data into an observation tensor."""
        obs_vec = np.concatenate((
            self.linear_vel_,
            self.angular_vel_,
            self.joint_pos_,
            self.joint_vel_,
            self.height_scan_big_,
            self.height_scan_small_,
        ), axis=0)

        return torch.from_numpy(obs_vec).float()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def timer_callback(self):
        """Build the observation tensor and log sensor state."""
        self.input_obs_tensor_ = self.build_observation()
        self._print_row('>>> TENSOR', self.input_obs_tensor_)

        self.get_logger().info(
            f'Obs: '
            f'linear_vel=({self.vx_:.2f},'
            f'{self.vy_:.2f},{self.vz_:.2f}), '
            f'angular_vel=({self.wx_:.2f},'
            f'{self.wy_:.2f},{self.wz_:.2f}), '
            f'jpos={len(self.joint_pos_)}, '
            f'h_big={self.height_scan_big_.shape}, '
            f'h_small={self.height_scan_small_.shape}, '
            f'full_obs={self.input_obs_tensor_.shape}',
        )
        self._print_row(
            '>>> big_elevation_map', self.height_scan_big_,
        )
        self._print_row(
            '>>> small_elevation_map', self.height_scan_small_,
        )


def main(args=None):
    """ROS2 entry point."""
    rclpy.init(args=args)
    node = WalkRLNetwork()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
