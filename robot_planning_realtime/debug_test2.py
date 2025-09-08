#!/usr/bin/env python3
"""Debug test with more detailed logging."""

import sys
import os
sys.path.append('../robot_kinematics/src')

import numpy as np
import logging
from robot_controller import RobotController
from robot_planning.realtime_planner import ProductionMotionPlanner, PlanningConstraints

# Enable debug logging to see everything
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def test_single_pose():
    """Test a single pose to see what happens."""
    print("🔧 Testing single pose with debug logging...")
    
    robot_controller = RobotController()
    constraints = PlanningConstraints(
        max_planning_time=2.0,
        max_ik_time_per_pose=0.5,
        position_tolerance=0.002,
        orientation_tolerance=0.005
    )
    planner = ProductionMotionPlanner(robot_controller, constraints)
    
    test_pose = np.array([0.4, -0.2, 0.6])
    print(f"Testing pose: {test_pose}")
    
    # Test only the validator
    pose_matrix = planner._ensure_pose_matrix(test_pose)
    
    print("\n🧪 Direct call to validator.is_pose_reachable...")
    try:
        result = planner.validator.is_pose_reachable(pose_matrix)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_pose()
