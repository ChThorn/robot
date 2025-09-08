#!/usr/bin/env python3
"""Debug joint limits structure."""

import sys
import os
sys.path.append('../robot_kinematics/src')

import numpy as np
import logging
from robot_controller import RobotController
from robot_planning.realtime_planner import ProductionMotionPlanner, PlanningConstraints

# Enable debug logging to see joint limits structure
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def debug_joint_limits():
    """Debug the joint limits structure."""
    print("🔧 Debugging joint limits...")
    
    robot_controller = RobotController()
    constraints = PlanningConstraints()
    planner = ProductionMotionPlanner(robot_controller, constraints)
    
    # Test a simple pose
    pose = np.array([0.3, 0.0, 0.5])
    pose_matrix = planner._ensure_pose_matrix(pose)
    
    # Try to trigger the joint limits validation
    result = planner.validator.is_pose_reachable(pose_matrix)
    print(f"Result: {result}")

if __name__ == "__main__":
    debug_joint_limits()
