"""
Robot Planning Library

A comprehensive motion planning library for robot manipulators using AORRTC algorithms.
"""

__version__ = "1.0.0"
__author__ = "Robot Planning Team"

# Import main classes for easy access
from .motion_planning import MotionPlanner, JointSpacePlanner, RobotEnvironment
from .path_planning import CartesianPathPlanner, Environment3D
from .trajectory_planning import TrajectoryPlanner, TrajectoryConstraints, TrajectoryInterpolator
from .utils import PlanningConfig, load_config

__all__ = [
    "MotionPlanner",
    "JointSpacePlanner", 
    "RobotEnvironment",
    "CartesianPathPlanner",
    "Environment3D",
    "TrajectoryPlanner",
    "TrajectoryConstraints",
    "TrajectoryInterpolator",
    "PlanningConfig",
    "load_config",
]

