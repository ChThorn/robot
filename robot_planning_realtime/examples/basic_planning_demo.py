#!/usr/bin/env python3
"""
Basic planning demonstration showing joint space and Cartesian planning.
"""

import numpy as np
import sys
import os
import logging

# Add robot_kinematics to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))

# Add robot_planning to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from robot_controller import RobotController
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Robot kinematics not available: {e}")
    print("This demo requires the robot_kinematics package.")
    ROBOT_AVAILABLE = False

if ROBOT_AVAILABLE:
    from robot_planning import MotionPlanner, TrajectoryPlanner
    from robot_planning.visualization import PlanningVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_joint_space_planning():
    """Demonstrate joint space planning."""
    if not ROBOT_AVAILABLE:
        print("Skipping joint space planning demo - robot kinematics not available")
        return
    
    print("\n" + "="*50)
    print("JOINT SPACE PLANNING DEMO")
    print("="*50)
    
    # Initialize robot controller
    robot_controller = RobotController()
    
    # Create motion planner
    motion_planner = MotionPlanner(robot_controller)
    
    # Define planning problem
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
    
    print(f"Start configuration: {start_config}")
    print(f"Goal configuration:  {goal_config}")
    
    # Plan path
    print("\nPlanning joint space path...")
    path = motion_planner.plan_joint_path(start_config, goal_config)
    
    if path is not None:
        print(f"✅ Path found with {len(path)} waypoints")
        
        # Compute path statistics
        path_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        print(f"Path length: {path_length:.3f} rad")
        
        # Validate path
        print("\nValidating path...")
        for i, config in enumerate(path):
            try:
                T = robot_controller.forward_kinematics(config)
                print(f"Waypoint {i}: FK successful, end-effector at {T[:3, 3]}")
            except Exception as e:
                print(f"❌ Waypoint {i}: FK failed - {e}")
                break
        else:
            print("✅ All waypoints validated successfully")
    else:
        print("❌ No path found")


def demo_trajectory_planning():
    """Demonstrate trajectory planning with time parameterization."""
    if not ROBOT_AVAILABLE:
        print("Skipping trajectory planning demo - robot kinematics not available")
        return
    
    print("\n" + "="*50)
    print("TRAJECTORY PLANNING DEMO")
    print("="*50)
    
    # Initialize components
    robot_controller = RobotController()
    from robot_planning.trajectory_planning import TrajectoryConstraints
    constraints = TrajectoryConstraints.default_constraints(6)
    trajectory_planner = TrajectoryPlanner(robot_controller, constraints)
    
    # Define planning problem
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([0.3, 0.2, -0.1, 0.05, 0.2, -0.05])
    
    print(f"Start configuration: {start_config}")
    print(f"Goal configuration:  {goal_config}")
    
    # Plan trajectory
    print("\nPlanning trajectory with trapezoidal velocity profile...")
    trajectory = trajectory_planner.plan_trajectory(
        start_config, goal_config,
        method='trapezoidal',
        interpolation='cubic'
    )
    
    if trajectory is not None:
        duration = trajectory.get_duration()
        print(f"✅ Trajectory planned successfully")
        print(f"Duration: {duration:.3f} seconds")
        
        # Sample trajectory at key points
        print("\nTrajectory samples:")
        sample_times = [0.0, duration/4, duration/2, 3*duration/4, duration]
        
        for t in sample_times:
            q = trajectory.evaluate(t)
            v = trajectory.evaluate_velocity(t)
            a = trajectory.evaluate_acceleration(t)
            
            print(f"t={t:.3f}s: q={q[:3]}, v={v[:3]}, a={a[:3]}")
        
        # Simulate execution
        print(f"\nSimulating trajectory execution...")
        success = trajectory_planner.execute_trajectory(trajectory, dry_run=True)
        
        if success:
            print("✅ Trajectory execution simulation successful")
        else:
            print("❌ Trajectory execution simulation failed")
    else:
        print("❌ Trajectory planning failed")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    if not ROBOT_AVAILABLE:
        print("Skipping visualization demo - robot kinematics not available")
        return
    
    print("\n" + "="*50)
    print("VISUALIZATION DEMO")
    print("="*50)
    
    # Initialize components
    robot_controller = RobotController()
    from robot_planning.trajectory_planning import TrajectoryConstraints
    constraints = TrajectoryConstraints.default_constraints(6)
    trajectory_planner = TrajectoryPlanner(robot_controller, constraints)
    visualizer = PlanningVisualizer(robot_controller)
    
    # Plan a simple trajectory
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([0.2, 0.1, -0.1, 0.0, 0.1, 0.0])
    
    print("Planning trajectory for visualization...")
    trajectory = trajectory_planner.plan_trajectory(
        start_config, goal_config,
        method='trapezoidal'
    )
    
    if trajectory is not None:
        print("Creating visualizations...")
        
        # Create joint trajectory plot
        fig1 = visualizer.plot_joint_trajectory(
            trajectory,
            save_path="joint_trajectory.html",
            show_velocity=True,
            show_acceleration=True
        )
        print("✅ Joint trajectory plot saved to 'joint_trajectory.html'")
        
        # Create Cartesian path visualization
        cartesian_path = []
        duration = trajectory.get_duration()
        times = np.linspace(0, duration, 50)
        
        for t in times:
            q = trajectory.evaluate(t)
            try:
                T = robot_controller.forward_kinematics(q)
                cartesian_path.append(T)
            except:
                pass
        
        if cartesian_path:
            fig2 = visualizer.plot_cartesian_path(
                cartesian_path,
                save_path="cartesian_path.html",
                show_orientation=False
            )
            print("✅ Cartesian path plot saved to 'cartesian_path.html'")
        
        print("\n📊 Open the HTML files in your browser to view the visualizations!")
    else:
        print("❌ Could not plan trajectory for visualization")


def main():
    """Run all demonstrations."""
    print("🤖 Robot Planning Library - Basic Demonstrations")
    print("This demo showcases the core capabilities of the robot planning library.")
    
    if not ROBOT_AVAILABLE:
        print("\n⚠️  WARNING: Robot kinematics not available.")
        print("Please ensure the robot_kinematics package is properly installed.")
        print("Some demonstrations will be skipped.")
    
    try:
        demo_joint_space_planning()
        demo_trajectory_planning()
        demo_visualization()
        
        print("\n" + "="*50)
        print("🎉 ALL DEMONSTRATIONS COMPLETED!")
        print("="*50)
        
        if ROBOT_AVAILABLE:
            print("\n📋 Summary:")
            print("- Joint space planning: Uses AORRTC for efficient path finding")
            print("- Trajectory planning: Adds time parameterization with velocity/acceleration limits")
            print("- Visualization: Interactive plots for analysis and debugging")
            print("\n💡 Next steps:")
            print("- Customize planning parameters in config/default_config.yaml")
            print("- Try different start/goal configurations")
            print("- Experiment with obstacle environments")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

