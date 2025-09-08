# Robot Planning Library

A comprehensive motion planning library for robot manipulators using AORRTC (Asymptotically Optimal Rapidly-exploring Random Tree Connect) algorithms.

## Features

- **AORRTC Planning**: State-of-the-art path planning with asymptotic optimality
- **Dual-Space Planning**: Both Cartesian and joint space planning capabilities
- **Trajectory Generation**: Time-parameterized trajectory generation with velocity/acceleration constraints
- **Visualization**: Interactive 3D visualization using Plotly
- **Robot Integration**: Seamless integration with robot kinematics systems
- **Comprehensive Testing**: Full test suite with validation against real robot data

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from robot_planning import MotionPlanner, TrajectoryPlanner
from robot_planning.visualization import PlanningVisualizer

# Initialize planner with your robot
planner = MotionPlanner(robot_controller)

# Plan a path
start_config = [0, 0, 0, 0, 0, 0]
goal_config = [1, 0.5, -0.5, 0, 0, 0]
path = planner.plan_joint_path(start_config, goal_config)

# Generate trajectory
trajectory_planner = TrajectoryPlanner(robot_controller)
trajectory = trajectory_planner.plan_trajectory(start_config, goal_config)

# Visualize
visualizer = PlanningVisualizer()
visualizer.plot_trajectory(trajectory)
```

## Architecture

- `robot_planning.motion_planning`: Core motion planning algorithms
- `robot_planning.path_planning`: Geometric path planning in Cartesian space
- `robot_planning.trajectory_planning`: Time parameterization and trajectory generation
- `robot_planning.visualization`: Interactive visualization tools
- `robot_planning.utils`: Utility functions and helpers

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=robot_planning --cov-report=html
```

## Examples

See the `examples/` directory for comprehensive usage examples:
- `basic_planning.py`: Basic motion planning example
- `trajectory_demo.py`: Trajectory generation and execution
- `visualization_demo.py`: Interactive visualization examples

## License

MIT License - see LICENSE file for details.

