#!/usr/bin/env python3
"""
Robot API Demo - Shows how to use robot units (degrees and millimeters)
Compatible with real robot API methods like move_joint() and move_blend_point()
"""

import numpy as np
import sys
import os
import logging

# Add robot_kinematics to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))

# Add robot_planning to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from robot_controller import RobotController
from robot_planning.motion_planning import MotionPlanner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_robot_api_interface():
    """Demonstrate robot API interface with degrees and millimeters."""
    print("\n" + "="*60)
    print("ROBOT API INTERFACE DEMO")
    print("Uses degrees for joints and millimeters for positions")
    print("="*60)
    
    # Initialize robot controller and motion planner
    robot_controller = RobotController()
    planner = MotionPlanner(robot_controller)
    
    # ==========================================================================
    # 1. Joint Space Planning with Robot Units (degrees)
    # ==========================================================================
    print("\n1. Joint Space Planning with Robot Units (degrees)")
    print("-" * 50)
    
    # Define joint angles in degrees (as used by real robot API)
    start_joints_deg = np.array([0.0, -10.0, 90.0, 0.0, 0.0, 0.0])   # degrees
    goal_joints_deg = np.array([30.0, -20.0, 100.0, 15.0, 10.0, 5.0]) # degrees
    
    print(f"Start joints: {start_joints_deg} degrees")
    print(f"Goal joints:  {goal_joints_deg} degrees")
    
    joint_path = planner.plan_joint_path_robot_units(start_joints_deg, goal_joints_deg)
    if joint_path:
        print(f"✓ Success: Planned {len(joint_path)} waypoints")
        print(f"  Start: {joint_path[0]} degrees")
        print(f"  End:   {joint_path[-1]} degrees")
    else:
        print("✗ Joint planning failed")
    
    # ==========================================================================
    # 2. Cartesian Space Planning with Robot Units (mm and degrees)
    # ==========================================================================
    print("\n2. Cartesian Space Planning with Robot Units (mm and degrees)")
    print("-" * 50)
    
    # Define poses in robot units: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    start_pose_robot = np.array([400.0, 0.0, 600.0, 0.0, 0.0, 0.0])    # mm and degrees
    goal_pose_robot = np.array([450.0, 50.0, 620.0, 10.0, 5.0, 15.0])  # mm and degrees
    
    print(f"Start pose: {start_pose_robot} [mm, mm, mm, deg, deg, deg]")
    print(f"Goal pose:  {goal_pose_robot} [mm, mm, mm, deg, deg, deg]")
    
    result = planner.plan_cartesian_path_robot_units(start_pose_robot, goal_pose_robot)
    if result:
        cartesian_path, joint_path = result
        print(f"✓ Success: Planned {len(cartesian_path)} cartesian waypoints, {len(joint_path)} joint waypoints")
        print(f"  Start cartesian: {cartesian_path[0]} [mm, mm, mm, deg, deg, deg]")
        print(f"  End cartesian:   {cartesian_path[-1]} [mm, mm, mm, deg, deg, deg]")
        print(f"  Start joints:    {joint_path[0]} degrees")
        print(f"  End joints:      {joint_path[-1]} degrees")
    else:
        print("✗ Cartesian planning failed")
    
    # ==========================================================================
    # 3. Robot API Compatible Methods
    # ==========================================================================
    print("\n3. Robot API Compatible Methods")
    print("-" * 50)
    
    # move_joint() API compatibility
    print("\n3a. move_joint_robot_api() - Compatible with robot's move_joint()")
    target_joints = np.array([15.0, -25.0, 95.0, 10.0, 5.0, 3.0])  # degrees
    current_joints = np.array([0.0, -10.0, 90.0, 0.0, 0.0, 0.0])   # degrees
    
    print(f"Current joints: {current_joints} degrees")
    print(f"Target joints:  {target_joints} degrees")
    
    path = planner.move_joint_robot_api(target_joints, current_joints)
    if path:
        print(f"✓ Success: move_joint_robot_api planned {len(path)} waypoints")
        print(f"  Path[0]: {path[0]} degrees")
        print(f"  Path[-1]: {path[-1]} degrees")
    else:
        print("✗ move_joint_robot_api failed")
    
    # move_blend_point() API compatibility
    print("\n3b. move_blend_point_robot_api() - Compatible with robot's move_blend_point()")
    target_pose = np.array([420.0, 30.0, 610.0, 5.0, 3.0, 8.0])    # mm and degrees
    current_pose = np.array([400.0, 0.0, 600.0, 0.0, 0.0, 0.0])    # mm and degrees
    
    print(f"Current pose: {current_pose} [mm, mm, mm, deg, deg, deg]")
    print(f"Target pose:  {target_pose} [mm, mm, mm, deg, deg, deg]")
    
    result = planner.move_blend_point_robot_api(target_pose, current_pose)
    if result:
        cartesian_path, joint_path = result
        print(f"✓ Success: move_blend_point_robot_api planned {len(cartesian_path)} cartesian, {len(joint_path)} joint waypoints")
        print(f"  Cartesian path[0]: {cartesian_path[0]} [mm, mm, mm, deg, deg, deg]")
        print(f"  Cartesian path[-1]: {cartesian_path[-1]} [mm, mm, mm, deg, deg, deg]")
        print(f"  Joint path[0]: {joint_path[0]} degrees")
        print(f"  Joint path[-1]: {joint_path[-1]} degrees")
    else:
        print("✗ move_blend_point_robot_api failed")
    
    # ==========================================================================
    # 4. Unit Conversion Summary
    # ==========================================================================
    print("\n4. Unit Conversion Summary")
    print("-" * 50)
    print("Robot API Methods (Input/Output):")
    print("  • plan_joint_path_robot_units(): degrees ↔ degrees")
    print("  • plan_cartesian_path_robot_units(): [mm, mm, mm, deg, deg, deg] ↔ [mm, mm, mm, deg, deg, deg]")
    print("  • move_joint_robot_api(): degrees ↔ degrees")
    print("  • move_blend_point_robot_api(): [mm, mm, mm, deg, deg, deg] ↔ [mm, mm, mm, deg, deg, deg]")
    print("")
    print("Internal Planning System:")
    print("  • Uses SI units: meters and radians")
    print("  • Automatic conversion handled internally")
    print("  • Original methods still available for advanced users")


if __name__ == "__main__":
    demo_robot_api_interface()
