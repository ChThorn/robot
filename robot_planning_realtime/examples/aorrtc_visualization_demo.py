#!/usr/bin/env python3
"""
AORRTC visualization demonstration showing the planning process.
"""

import numpy as np
import sys
import os
import logging

# Add robot_kinematics to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from robot_controller import RobotController
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Robot kinematics not available: {e}")
    ROBOT_AVAILABLE = False

# Import the standalone AORRTC implementation for visualization
try:
    from robot_planning.aorrtc_smoothed import AORRTC3D, Environment3D as StandaloneEnvironment3D
    AORRTC_AVAILABLE = True
except ImportError as e:
    print(f"Standalone AORRTC not available: {e}")
    AORRTC_AVAILABLE = False

if ROBOT_AVAILABLE:
    from robot_planning import MotionPlanner
    from robot_planning.visualization import PlanningVisualizer
    from robot_planning.path_planning import Environment3D

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_standalone_aorrtc():
    """Demonstrate standalone AORRTC with visualization."""
    if not AORRTC_AVAILABLE:
        print("Skipping standalone AORRTC demo - implementation not available")
        return
    
    print("\n" + "="*60)
    print("STANDALONE AORRTC 3D VISUALIZATION DEMO")
    print("="*60)
    
    # Create environment with obstacles
    env = StandaloneEnvironment3D(bounds=((-2, 2), (-2, 2), (0, 2)))
    
    # Add some interesting obstacles
    env.add_sphere_obstacle([0.0, 0.0, 0.8], 0.3)
    env.add_sphere_obstacle([0.5, 0.5, 0.5], 0.25)
    env.add_sphere_obstacle([-0.5, -0.5, 1.2], 0.2)
    env.add_sphere_obstacle([0.8, -0.3, 0.3], 0.15)
    env.add_sphere_obstacle([-0.3, 0.8, 1.5], 0.18)
    
    print(f"Environment created with {len(env.obstacles)} obstacles")
    
    # Define planning problem
    start_point = np.array([-1.5, -1.5, 0.2])
    goal_point = np.array([1.5, 1.5, 1.8])
    
    print(f"Start: {start_point}")
    print(f"Goal:  {goal_point}")
    
    # Create planner
    planner = AORRTC3D(
        env, start_point, goal_point,
        max_iter=8000,
        step_size=0.15,
        goal_bias=0.1,
        connect_threshold=0.3,
        rewire_radius=0.5
    )
    
    # Plan path
    print("\nRunning AORRTC planning...")
    original_path = planner.plan()
    
    if original_path:
        print(f"✅ Path found with {len(original_path)} waypoints")
        
        # Smooth the path
        print("Smoothing path...")
        smoothed_path = planner.smooth_path_optimized(
            original_path,
            iterations=150,
            alpha=0.1,
            beta=0.6,
            gamma=0.2
        )
        
        # Create visualization
        print("Creating interactive visualization...")
        planner.visualize(
            original_path=original_path,
            smoothed_path=smoothed_path,
            save_html=True,
            show_in_browser=False
        )
        
        print("✅ Visualization saved as HTML file")
        print("📊 Open the generated HTML file in your browser to view the interactive 3D plot!")
        
        # Print path statistics
        original_cost = planner._path_cost(original_path)
        smoothed_cost = planner._path_cost(smoothed_path)
        improvement = (original_cost - smoothed_cost) / original_cost * 100
        
        print(f"\n📈 Path Statistics:")
        print(f"Original path cost:  {original_cost:.3f}")
        print(f"Smoothed path cost:  {smoothed_cost:.3f}")
        print(f"Improvement:         {improvement:.1f}%")
        
    else:
        print("❌ No path found")


def demo_integrated_aorrtc():
    """Demonstrate integrated AORRTC with robot kinematics."""
    if not ROBOT_AVAILABLE:
        print("Skipping integrated AORRTC demo - robot kinematics not available")
        return
    
    print("\n" + "="*60)
    print("INTEGRATED AORRTC WITH ROBOT KINEMATICS DEMO")
    print("="*60)
    
    # Initialize robot controller
    robot_controller = RobotController()
    motion_planner = MotionPlanner(robot_controller)
    visualizer = PlanningVisualizer(robot_controller)
    
    # Create environment with obstacles
    environment = Environment3D(workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)))
    environment.add_sphere_obstacle([0.2, 0.2, 0.8], 0.15)
    environment.add_box_obstacle([-0.3, -0.3, 0.5], [-0.1, -0.1, 0.7])
    
    print(f"Environment created with obstacles")
    
    # Define Cartesian planning problem
    start_pose = np.eye(4)
    start_pose[:3, 3] = [0.3, -0.3, 0.6]
    
    goal_pose = np.eye(4)
    goal_pose[:3, 3] = [-0.3, 0.3, 1.2]
    
    print(f"Start pose: {start_pose[:3, 3]}")
    print(f"Goal pose:  {goal_pose[:3, 3]}")
    
    # Plan Cartesian path
    print("\nPlanning Cartesian path with AORRTC...")
    cartesian_path = motion_planner.plan_cartesian_path(
        start_pose, goal_pose
    )
    
    if cartesian_path:
        print(f"✅ Cartesian path found with {len(cartesian_path)} waypoints")
        
        # Visualize the path
        print("Creating Cartesian path visualization...")
        fig = visualizer.plot_cartesian_path(
            cartesian_path,
            environment=environment,
            save_path="cartesian_aorrtc_path.html",
            show_orientation=True
        )
        
        print("✅ Cartesian path visualization saved to 'cartesian_aorrtc_path.html'")
        
        # Convert to joint space
        print("Converting to joint space path...")
        joint_path = motion_planner._cartesian_to_joint_path(cartesian_path)
        
        if joint_path:
            print(f"✅ Joint space path created with {len(joint_path)} configurations")
            
            # Compute path statistics
            joint_path_length = sum(
                np.linalg.norm(joint_path[i+1] - joint_path[i]) 
                for i in range(len(joint_path)-1)
            )
            print(f"Joint space path length: {joint_path_length:.3f} rad")
        else:
            print("❌ Failed to convert to joint space")
    else:
        print("❌ No Cartesian path found")


def demo_joint_space_aorrtc():
    """Demonstrate joint space AORRTC planning."""
    if not ROBOT_AVAILABLE:
        print("Skipping joint space AORRTC demo - robot kinematics not available")
        return
    
    print("\n" + "="*60)
    print("JOINT SPACE AORRTC PLANNING DEMO")
    print("="*60)
    
    # Initialize components
    robot_controller = RobotController()
    motion_planner = MotionPlanner(robot_controller)
    
    # Define challenging joint space problem
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([1.2, 0.8, -0.6, 0.4, 1.0, -0.3])
    
    print(f"Start configuration: {start_config}")
    print(f"Goal configuration:  {goal_config}")
    
    # Plan with joint space AORRTC
    print("\nPlanning with joint space AORRTC...")
    joint_path = motion_planner.plan_joint_path(start_config, goal_config)
    
    if joint_path:
        print(f"✅ Joint space path found with {len(joint_path)} waypoints")
        
        # Compute forward kinematics for visualization
        cartesian_waypoints = []
        for config in joint_path[::5]:  # Subsample for visualization
            try:
                T = robot_controller.forward_kinematics(config)
                cartesian_waypoints.append(T[:3, 3])
            except Exception as e:
                print(f"FK failed for config {config}: {e}")
        
        if cartesian_waypoints:
            print(f"Computed {len(cartesian_waypoints)} Cartesian waypoints")
            
            # Create simple visualization of end-effector path
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            waypoints = np.array(cartesian_waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   'b-', linewidth=2, label='End-effector path')
            ax.scatter(waypoints[0, 0], waypoints[0, 1], waypoints[0, 2], 
                      c='green', s=100, label='Start')
            ax.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], 
                      c='red', s=100, label='Goal')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Joint Space AORRTC - End-Effector Path')
            ax.legend()
            
            plt.savefig('joint_space_aorrtc_path.png', dpi=150, bbox_inches='tight')
            print("✅ End-effector path plot saved to 'joint_space_aorrtc_path.png'")
            plt.close()
        
        # Compute path statistics
        path_length = sum(
            np.linalg.norm(joint_path[i+1] - joint_path[i]) 
            for i in range(len(joint_path)-1)
        )
        print(f"Joint space path length: {path_length:.3f} rad")
        
        # Validate all configurations
        print("Validating all joint configurations...")
        valid_count = 0
        for i, config in enumerate(joint_path):
            try:
                T = robot_controller.forward_kinematics(config)
                valid_count += 1
            except Exception as e:
                print(f"❌ Configuration {i} invalid: {e}")
        
        print(f"✅ {valid_count}/{len(joint_path)} configurations validated successfully")
        
    else:
        print("❌ No joint space path found")


def main():
    """Run all AORRTC visualization demonstrations."""
    print("🤖 Robot Planning Library - AORRTC Visualization Demonstrations")
    print("This demo showcases the AORRTC algorithm with interactive visualizations.")
    
    try:
        demo_standalone_aorrtc()
        demo_integrated_aorrtc()
        demo_joint_space_aorrtc()
        
        print("\n" + "="*60)
        print("🎉 ALL AORRTC DEMONSTRATIONS COMPLETED!")
        print("="*60)
        
        print("\n📋 Generated Files:")
        print("- HTML files: Interactive 3D visualizations (open in browser)")
        print("- PNG files: Static plots for documentation")
        
        print("\n💡 Key Features Demonstrated:")
        print("- Standalone AORRTC: Pure geometric planning with obstacles")
        print("- Integrated AORRTC: Planning with robot kinematics constraints")
        print("- Joint space AORRTC: Direct planning in configuration space")
        print("- Path smoothing: Optimization-based trajectory refinement")
        print("- Interactive visualization: 3D plots with Plotly")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

