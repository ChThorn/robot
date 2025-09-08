#!/usr/bin/env python3
"""Test to find poses that ARE reachable."""

import sys
import os
sys.path.append('../robot_kinematics/src')

import numpy as np
import logging
from robot_controller import RobotController
from robot_planning.realtime_planner import ProductionMotionPlanner, PlanningConstraints

# Disable most logging to see results clearly
logging.basicConfig(level=logging.ERROR, format='%(levelname)s:%(name)s:%(message)s')

def find_reachable_poses():
    """Find some poses that are reachable."""
    print("🔍 Finding reachable poses...")
    
    robot_controller = RobotController()
    constraints = PlanningConstraints(
        max_planning_time=2.0,
        max_ik_time_per_pose=0.5,
        position_tolerance=0.002,
        orientation_tolerance=0.005
    )
    planner = ProductionMotionPlanner(robot_controller, constraints)
    
    # Test some conservative poses that should be reachable
    test_poses = [
        np.array([0.3, 0.0, 0.5]),   # Conservative forward reach
        np.array([0.2, 0.0, 0.4]),   # Even more conservative
        np.array([0.5, 0.0, 0.3]),   # Wide but low
        np.array([0.0, 0.0, 0.6]),   # Straight up
        np.array([0.0, 0.3, 0.5]),   # To the side
    ]
    
    reachable_poses = []
    
    for i, pose in enumerate(test_poses):
        print(f"\nTesting pose {i+1}: {pose}")
        
        try:
            pose_matrix = planner._ensure_pose_matrix(pose)
            result = planner.validator.is_pose_reachable(pose_matrix)
            reachable, q_solution = result
            
            if reachable:
                print(f"  ✅ REACHABLE! Joint solution found.")
                reachable_poses.append(pose)
            else:
                print(f"  ❌ Not reachable")
                
        except Exception as e:
            print(f"  💥 Error: {e}")
    
    print(f"\n🎯 Found {len(reachable_poses)} reachable poses:")
    for pose in reachable_poses:
        print(f"  {pose}")
    
    return reachable_poses

if __name__ == "__main__":
    find_reachable_poses()
