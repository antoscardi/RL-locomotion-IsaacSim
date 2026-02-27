# Humanoid RL Locomotion

ROS 2 node for real-time reinforcement-learning-based locomotion control of the **Unitree G1 (29-DoF)** humanoid robot, trained in NVIDIA IsaacLab.

## Overview

The `walk_rl_network` node collects sensor data from the robot, assembles an observation tensor, and runs inference on a pre-trained policy network to produce joint position commands for walking — including operation while carrying a welder payload.

### Observation Space

| Component | Dimensions | Source Topic |
|---|:---:|---|
| Base angular velocity | 3 | `/odom_robot` |
| Projected gravity | 3 | `/odom_robot` |
| Velocity commands | 3 | `/cmd_vel` |
| Joint positions (relative) | 29 | `/joint_states` |
| Joint velocities (relative) | 29 | `/joint_states` |
| Last action | 12 | Network output (t−1) |
| **History buffer** | **×5** | — |

Total policy input: **395** (79 × 5 history steps)

### Action Space

12 joint position targets (hip, knee, ankle) at 50 Hz (`step_dt = 0.02 s`).

### Velocity Limits

| | Min | Max |
|---|:---:|:---:|
| Linear X (m/s) | −0.5 | 1.0 |
| Linear Y (m/s) | −0.3 | 0.3 |
| Angular Z (rad/s) | −0.2 | 0.2 |

## Project Structure

```
humanoid_rl_locomotion/
├── humanoid_rl_locomotion/
│   ├── __init__.py
│   └── walk_rl_network.py          # ROS 2 node
├── model/
│   ├── walking_with_welder.pt       # Exported policy (JIT)
│   └── walking_with_welder/
│       ├── model_11900.pt           # Checkpoint at step 11900
│       └── params/
│           ├── agent.yaml           # RSL-RL agent config
│           ├── deploy.yaml          # Deployment / joint mapping config
│           ├── env.yaml             # Full environment config
│           └── velocity_env_cfg.py  # IsaacLab env definition
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## Dependencies

- **ROS 2 Humble**
- Python 3.10+
- PyTorch ≥ 2.0
- NumPy

### ROS 2 Message Packages

- `nav_msgs` — `Odometry`
- `sensor_msgs` — `JointState`
- `geometry_msgs` — `Twist`
- `grid_map_msgs` — `GridMap`

## Build & Run

```bash
# Build
cd <workspace_root>
colcon build --packages-select humanoid_rl_locomotion

# Source
source install/setup.bash

# Run
ros2 run humanoid_rl_locomotion walk_rl_network
```

## Training

The policy was trained using [IsaacLab](https://github.com/isaac-sim/IsaacLab) with the Unitree G1 29-DoF configuration. The environment definition is in `model/walking_with_welder/params/velocity_env_cfg.py`. Key training features:

- Domain randomisation (friction, mass, external pushes)
- Curriculum-based terrain generation (flat, slopes, waves, rough)
- Gait rewards with 0.8 s period
- Welder + pistol payload mass randomisation

## License

See `package.xml` for license information.
