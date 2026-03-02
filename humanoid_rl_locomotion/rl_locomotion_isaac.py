#!/usr/bin/env python3
"""ROS2 node for deploying a pretrained IsaacLab policy to IsaacSim.

Subscribes to observation topics exported by the ROS2–IsaacSim bridge
(odometry, joint states) and publishes joint-position commands computed
by a PyTorch ActorCritic policy trained with PPO in IsaacLab.

Observations are assembled into a 5-timestep history buffer (395 dims)
as expected by the trained network.

Usage::

    ros2 run humanoid_rl_locomotion rl_locomotion_isaac \\
        --ros-args -p model_path:="/path/to/model.pt" \\
                   -p config_path:="/path/to/deploy.yaml"
"""

import os
from collections import deque

import numpy as np
import torch
import yaml

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ActorCriticPolicy(torch.nn.Module):
    """Actor network from an IsaacLab ActorCritic PPO policy.

    Implements the MLP used by the actor head with numbered layer
    attributes matching the checkpoint key format
    (e.g. ``0.weight``, ``1.bias``, ``2.weight``).

    Parameters
    ----------
    observation_size : int
        Dimension of the observation feature vector.
    num_actions : int
        Number of actuated joint outputs.
    hidden_layer_sizes : list[int]
        Width of each hidden layer.
    activation : str
        Activation function name (``elu``, ``relu``, ``tanh``).
    """

    SUPPORTED_ACTIVATIONS = {
        'elu': torch.nn.ELU,
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh,
    }

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_layer_sizes: list,
        activation: str = 'elu',
    ):
        super().__init__()
        activation_class = self.SUPPORTED_ACTIVATIONS.get(
            activation, torch.nn.ELU,
        )

        layer_sizes = [observation_size] + hidden_layer_sizes + [num_actions]
        layer_index = 0
        self._sequential_layer_ids: list[int] = []

        for i in range(len(layer_sizes) - 1):
            linear_layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            setattr(self, str(layer_index), linear_layer)
            self._sequential_layer_ids.append(layer_index)
            layer_index += 1
            if i < len(layer_sizes) - 2:
                setattr(self, str(layer_index), activation_class())
                self._sequential_layer_ids.append(layer_index)
                layer_index += 1

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        output = observation
        for layer_id in self._sequential_layer_ids:
            output = getattr(self, str(layer_id))(output)
        return output


class LocomotionPolicyNode(Node):
    """ROS2 node: policy inference at 10 Hz."""

    # from env.yaml
    NUM_JOINTS = 29
    NUM_ACTIONS = 12

    # from deploy.yaml
    HISTORY_LENGTH = 5
    SINGLE_OBS_SIZE = 67
    FULL_OBS_SIZE = (SINGLE_OBS_SIZE + NUM_ACTIONS) * HISTORY_LENGTH
    CONTROL_RATE_SEC = 0.1
    ANGULAR_VELOCITY_SCALE = 0.2
    JOINT_VELOCITY_SCALE = 0.05

    def __init__(self):
        super().__init__('rl_locomotion_isaac')

        self.declare_parameter('model_path', '')
        self.declare_parameter('config_path', '')
        model_path = (
            self.get_parameter('model_path')
            .get_parameter_value()
            .string_value
        )
        config_path = (
            self.get_parameter('config_path')
            .get_parameter_value()
            .string_value
        )

        if not model_path:
            self.get_logger().error('model_path parameter is required')
            raise RuntimeError('model_path not provided')

        if config_path:
            self.deploy_config = self._load_deploy_config(config_path)
        else:
            self.deploy_config = {}

        self.policy_network = self._load_policy_network(model_path)
        self.policy_network.eval()
        self.get_logger().info(f'Loaded policy from {model_path}')

        self.angular_vel = np.zeros(3, dtype=np.float32)
        self.joint_pos = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        self.joint_vel = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        self.vel_command = np.zeros(3, dtype=np.float32)

        self.observation_history: deque[np.ndarray] = deque(
            maxlen=self.HISTORY_LENGTH,
        )
        self.action_history: deque[np.ndarray] = deque(
            maxlen=self.HISTORY_LENGTH,
        )

        initial_observation = self._build_single_observation()
        for _ in range(self.HISTORY_LENGTH):
            self.observation_history.append(
                initial_observation.copy(),
            )
            self.action_history.append(
                np.zeros(self.NUM_ACTIONS, dtype=np.float32),
            )

        # from deploy.yaml → actions.JointPositionAction
        joint_action_config = self.deploy_config.get(
            'actions', {},
        ).get('JointPositionAction', {})
        self.action_scale = np.array(
            joint_action_config.get(
                'scale', [0.25] * self.NUM_ACTIONS,
            ),
            dtype=np.float32,
        )
        self.action_offset = np.array(
            joint_action_config.get(
                'offset',
                [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0,
                 0.3, 0.3, -0.2, -0.2, 0.0, 0.0],
            ),
            dtype=np.float32,
        )

        self.get_logger().info(f'Action scale:  {self.action_scale}')
        self.get_logger().info(f'Action offset: {self.action_offset}')

        self.create_subscription(
            Odometry, '/isaac/odom', self._odom_cb, 10)
        self.create_subscription(
            JointState, '/isaac/joint_states', self._joint_state_cb, 10)
        self.command_pub = self.create_publisher(
            JointState, '/isaac/joint_commands', 10)

        self.create_timer(self.CONTROL_RATE_SEC, self._run_policy_step)

    @staticmethod
    def _load_deploy_config(filepath: str) -> dict:
        if not os.path.isfile(filepath):
            return {}
        try:
            with open(filepath, 'r') as config_file:
                return yaml.safe_load(config_file) or {}
        except Exception:
            return {}

    def _load_policy_network(
        self, checkpoint_path: str,
    ) -> ActorCriticPolicy:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path, map_location='cpu',
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'actor_state_dict' in checkpoint:
                state_dict = checkpoint['actor_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        actor_weights = {
            key.replace('actor.', ''): tensor
            for key, tensor in state_dict.items()
            if key.startswith('actor.')
        }

        if '0.weight' in actor_weights:
            observation_size = int(
                actor_weights['0.weight'].shape[1],
            )
        else:
            observation_size = self.FULL_OBS_SIZE

        network = ActorCriticPolicy(
            observation_size, self.NUM_ACTIONS, [512, 256, 128],
        )
        network.load_state_dict(actor_weights, strict=False)
        return network

    def _odom_cb(self, msg: Odometry):
        twist = msg.twist.twist
        self.angular_vel[:] = [
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ]

    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) >= self.NUM_JOINTS:
            self.joint_pos[:] = msg.position[:self.NUM_JOINTS]
        if len(msg.velocity) >= self.NUM_JOINTS:
            self.joint_vel[:] = msg.velocity[:self.NUM_JOINTS]

    def _build_single_observation(self) -> np.ndarray:
        """Assemble a single-timestep observation (67 dims).

        Layout (from deploy.yaml):
          base_ang_vel       × 0.2   →  3
          projected_gravity  × 1.0   →  3
          velocity_commands  × 1.0   →  3
          joint_pos_rel      × 1.0   → 29
          joint_vel_rel      × 0.05  → 29
        """
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        single_observation = np.concatenate([
            self.angular_vel * self.ANGULAR_VELOCITY_SCALE,
            projected_gravity,
            self.vel_command,
            self.joint_pos,
            self.joint_vel * self.JOINT_VELOCITY_SCALE,
        ])
        return single_observation

    def _build_observation_history_vector(self) -> torch.Tensor:
        """Concatenate 5 history frames into a 395-dim policy input.

        Structure: [obs_t−4, act_t−4, …, obs_t, act_t]
        (67 + 12) × 5 = 395
        """
        self.observation_history.append(self._build_single_observation())

        history_parts = []
        for observation, previous_action in zip(
            self.observation_history, self.action_history,
        ):
            history_parts.append(observation)
            history_parts.append(previous_action)

        return torch.from_numpy(
            np.concatenate(history_parts),
        ).float()

    def _run_policy_step(self):
        full_observation = self._build_observation_history_vector()

        try:
            with torch.no_grad():
                raw_action = self.policy_network(
                    full_observation.unsqueeze(0),
                )
        except RuntimeError as exc:
            self.get_logger().error(f'Inference failed: {exc}')
            return

        raw_action = raw_action.squeeze(0).cpu().numpy()

        scaled_action = (
            raw_action * self.action_scale + self.action_offset
        )
        self.action_history.append(scaled_action.copy())

        joint_command_msg = JointState()
        joint_command_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        joint_command_msg.position = scaled_action.tolist()
        self.command_pub.publish(joint_command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocomotionPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
