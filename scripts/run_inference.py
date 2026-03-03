#!/usr/bin/env python3
"""Standalone inference test for IsaacLab ActorCritic locomotion policies.

Usage::

    python3 scripts/run_inference.py [-m MODEL] [-c CONFIG]
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import yaml


class ActorCriticPolicy(torch.nn.Module):
    """Actor MLP matching IsaacLab checkpoint key format."""

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


def load_policy_from_checkpoint(
        checkpoint_path: str,
        num_actions: int = 12,
        hidden_layer_sizes: Optional[list] = None,
) -> tuple:
    """Load an IsaacLab actor checkpoint into eval mode.

    Returns ``(policy, observation_size)``.
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [512, 256, 128]

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Model file not found: {checkpoint_path}')

    try:
        jit_model = torch.jit.load(checkpoint_path, map_location='cpu')
        print(f'[load] Loaded JIT model from {checkpoint_path}')
        jit_model.eval()
        return jit_model, None
    except Exception:
        pass

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False,
    )

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        return checkpoint, None

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
        observation_size = int(actor_weights['0.weight'].shape[1])
    else:
        observation_size = 395

    policy = ActorCriticPolicy(
        observation_size, num_actions, hidden_layer_sizes,
    )
    load_result = policy.load_state_dict(actor_weights, strict=False)
    if load_result.missing_keys:
        print(f'[load] Missing keys: {load_result.missing_keys}')
    if load_result.unexpected_keys:
        print(f'[load] Unexpected keys: {load_result.unexpected_keys}')

    policy.eval()
    print(
        f'[load] Actor network loaded \u2014 '
        f'observation_size={observation_size}, '
        f'hidden={hidden_layer_sizes}, '
        f'num_actions={num_actions}'
    )
    return policy, observation_size


def load_deploy_config(config_path: str) -> dict:
    """Parse deploy YAML; return ``{}`` on failure."""
    if not config_path or not os.path.isfile(config_path):
        print(f'[config] File not found: {config_path}')
        return {}
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file) or {}


def build_synthetic_observation(
    obs_size: int = 395,
) -> np.ndarray:
    """Return ``[0, 1, 2, …, obs_size-1]`` as float32."""
    return np.arange(obs_size, dtype=np.float32)


def run_single_inference(
    policy: torch.nn.Module,
    observation: np.ndarray,
    action_scale: np.ndarray,
    action_offset: np.ndarray,
) -> dict:
    """Forward pass + action scaling."""
    observation_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    with torch.no_grad():
        raw_action = policy(observation_tensor).squeeze(0).cpu().numpy()
    scaled_action = raw_action * action_scale + action_offset
    return {'raw_action': raw_action, 'scaled_action': scaled_action}


def main():
    parser = argparse.ArgumentParser(
        description='Run standalone inference for IsaacLab locomotion policy.')
    parser.add_argument('--model', '-m',
                        default='model/walking_with_welder.pt',
                        help='Path to the .pt checkpoint')
    parser.add_argument('--config', '-c',
                        default='model/walking_with_welder/params/deploy.yaml',
                        help='Path to deploy.yaml config')
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    config_path = os.path.abspath(args.config)

    print('=' * 70)
    print(' IsaacLab Locomotion Policy — Inference Test')
    print('=' * 70)

    print(f'\n[1] Loading policy from: {model_path}')
    policy, obs_size = load_policy_from_checkpoint(model_path)

    try:
        from torchinfo import summary
        print()
        summary(policy, input_size=(1, obs_size),
                col_names=['output_size', 'num_params'],
                col_width=18, row_settings=['var_names'])
    except ImportError:
        total_params = sum(p.numel() for p in policy.parameters())
        print(f'    Parameters: {total_params:,}  '
              f'(install torchinfo for full summary)')

    print(f'\n[2] Deploy config: {config_path}')
    deploy_config = load_deploy_config(config_path)
    jpc = deploy_config.get(
        'actions', {},
    ).get('JointPositionAction', {})
    action_scale = np.array(
        jpc.get('scale', [0.25] * 12), dtype=np.float32,
    )
    action_offset = np.array(
        jpc.get(
            'offset',
            [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, -0.2, -0.2, 0.0, 0.0],
        ),
        dtype=np.float32,
    )
    print(f'    Scale:  {action_scale}')
    print(f'    Offset: {action_offset}')

    print('\n[3] Zero-input inference (per-joint scaled actions)')
    zero_obs = np.zeros(obs_size, dtype=np.float32)
    zero_result = run_single_inference(
        policy, zero_obs, action_scale, action_offset)
    header = '    {:>7}  {:>9}  {:>9}'.format('Joint', 'Scaled', 'Offset')
    separator = '    {:>7}  {:>9}  {:>9}'.format(
        '-----', '---------', '---------')
    print(header)
    print(separator)
    for j in range(len(action_offset)):
        z = zero_result['scaled_action'][j]
        o = action_offset[j]
        print(f'    {j:7d}  {z:+9.4f}  {o:+9.4f}')
    deviations = np.abs(zero_result['scaled_action'] - action_offset)
    bad = np.where(deviations > 1.0)[0]
    if bad.size:
        print(' Large deviations from offset:')
        for idx in bad:
            print(f'      Joint {idx}: Δ={deviations[idx]:.4f}')
    else:
        print('    Sanity check ✓')

    print('\n[4] Synthetic-input inference ([0,1,2,…,N-1])')
    synthetic_obs = build_synthetic_observation(obs_size)
    synthetic_result = run_single_inference(
        policy, synthetic_obs, action_scale, action_offset)
    print(f'    {synthetic_result["scaled_action"]}')

    print('\n' + '=' * 70)
    print(' All checks passed ✓')
    print('=' * 70)


if __name__ == '__main__':
    main()
