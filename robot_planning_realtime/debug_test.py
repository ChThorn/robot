#!/usr/bin/env python3
"""Debug test to isolate the unpacking issue."""

import sys
import os
sys.path.append('../robot_kinematics/src')

import numpy as np
import logging
from robot_controller import RobotController
from robot_planning.realtime_planner import ProductionMotionPlanner, PlanningConstraints

# Enable debug logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

def test_ik_directly():
    """Test IK validation directly to isolate the issue."""
    print("🔧 Testing IK validation directly...")
    
    # Initialize components
    robot_controller = RobotController()
    constraints = PlanningConstraints(
        max_planning_time=2.0,
        max_ik_time_per_pose=0.5,
        position_tolerance=0.002,
        orientation_tolerance=0.005
    )
    planner = ProductionMotionPlanner(robot_controller, constraints)
    
    # Test pose that was working in the demo before the error
    test_pose = np.array([0.4, -0.2, 0.6])
    print(f"Testing pose: {test_pose}")
    
    # Convert to 4x4 matrix
    pose_matrix = planner._ensure_pose_matrix(test_pose)
    print(f"Converted to matrix shape: {pose_matrix.shape}")
    print(f"Matrix:\n{pose_matrix}")
    
    # Test direct IK validation
    try:
        print("\n🧪 Testing validator.is_pose_reachable...")
        result = planner.validator.is_pose_reachable(pose_matrix)
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        print(f"Result: {result}")
        
        # Try unpacking
        reachable, q_solution = result
        print(f"✅ Unpacked successfully: reachable={reachable}, q_solution={q_solution is not None}")
        
    except Exception as e:
        print(f"❌ Error in is_pose_reachable: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ik_directly()
