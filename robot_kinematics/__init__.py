"""
Robot Kinematics Package for 6-DOF Manipulator

A production-ready kinematics library for 6-DOF robot manipulators using
Product of Exponentials (PoE) formulation with comprehensive validation.

Author: Robot Control Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Robot Control Team"

from .src.robot_kinematics import RobotKinematics, RobotKinematicsError
from .src.robot_controller import RobotController
from .src.config import KinematicsConfig, get_config, set_config_file
from .src.kinematics_validation import KinematicsValidator, run_comprehensive_validation

__all__ = [
    'RobotKinematics',
    'RobotKinematicsError', 
    'RobotController',
    'KinematicsConfig',
    'get_config',
    'set_config_file',
    'KinematicsValidator',
    'run_comprehensive_validation'
]

