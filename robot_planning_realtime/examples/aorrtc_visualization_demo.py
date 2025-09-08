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
    
    # Set more conservative IK parameters to prevent long running times
    # IK parameters for path planning (production-grade precision)
    robot_controller.set_ik_parameters(
        max_iters=300,      # Production-grade iterations
        num_attempts=10,    # More attempts for better convergence
        pos_tol=0.002,      # Production tolerance: 2mm
        rot_tol=0.005       # Production tolerance: 0.29 degrees
    )
    
    # Create environment with obstacles in robot's reachable workspace
    environment = Environment3D(workspace_bounds=((-0.8, 0.8), (-0.8, 0.8), (0.2, 1.0)))
    environment.add_sphere_obstacle([0.1, 0.1, 0.6], 0.08)
    environment.add_box_obstacle([-0.2, -0.2, 0.4], [-0.1, -0.1, 0.5])
    
    print(f"Environment created with obstacles")
    
    # Define Cartesian planning problem with reachable poses
    start_pose = np.eye(4)
    start_pose[:3, 3] = [0.4, -0.2, 0.6]  # Reachable pose
    
    goal_pose = np.eye(4)
    goal_pose[:3, 3] = [-0.2, 0.4, 0.8]   # Reachable pose
    
    print(f"Start pose: {start_pose[:3, 3]}")
    print(f"Goal pose:  {goal_pose[:3, 3]}")
    
    # Validate that poses are reachable before planning
    print("\nValidating workspace reachability...")
    try:
        start_q, start_converged = robot_controller.inverse_kinematics(start_pose)
        goal_q, goal_converged = robot_controller.inverse_kinematics(goal_pose)
        
        if not start_converged:
            print("❌ Start pose not reachable, using default pose")
            start_pose[:3, 3] = [0.3, 0.0, 0.7]
            start_q, start_converged = robot_controller.inverse_kinematics(start_pose)
            
        if not goal_converged:
            print("❌ Goal pose not reachable, using default pose") 
            goal_pose[:3, 3] = [-0.3, 0.0, 0.7]
            goal_q, goal_converged = robot_controller.inverse_kinematics(goal_pose)
            
        if start_converged and goal_converged:
            print("✅ Both poses are reachable")
        else:
            print("❌ Unable to find reachable poses, skipping demo")
            return
            
    except Exception as e:
        print(f"❌ Workspace validation failed: {e}")
        return
    
    # Plan Cartesian path with timeout
    print("\nPlanning Cartesian path with AORRTC...")
    print(f"Final start pose used: {start_pose[:3, 3]}")
    print(f"Final goal pose used:  {goal_pose[:3, 3]}")
    import time
    start_time = time.time()
    result = motion_planner.plan_cartesian_path(
        start_pose, goal_pose
    )
    planning_time = time.time() - start_time
    
    if result:
        cartesian_path, joint_path = result
        print(f"✅ Cartesian path found in {planning_time:.2f}s with {len(cartesian_path)} cartesian waypoints, {len(joint_path)} joint waypoints")
        
        # Debug: Print the actual path waypoints
        print("\n🔍 Path waypoints analysis:")
        for i, waypoint in enumerate(cartesian_path):
            if len(waypoint.shape) == 2 and waypoint.shape == (4, 4):
                pos = waypoint[:3, 3]
            else:
                pos = waypoint[:3] if len(waypoint) >= 3 else waypoint
            print(f"  Waypoint {i}: [{pos[0]:8.6f}, {pos[1]:8.6f}, {pos[2]:8.6f}]")
        
        # Visualize the path
        print("Creating Cartesian path visualization...")
        try:
            fig = visualizer.plot_cartesian_path(
                cartesian_path,
                environment=environment,
                save_path="cartesian_aorrtc_path.html",
                show_orientation=True
            )
            print("✅ Cartesian path visualization saved to 'cartesian_aorrtc_path.html'")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
        
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
        print(f"❌ No Cartesian path found (took {planning_time:.2f}s)")
        
    # Reset IK parameters to production-grade defaults for final validation
    robot_controller.set_ik_parameters(
        max_iters=500,      # High precision for final validation
        num_attempts=50,    # Extensive attempts for critical operations
        pos_tol=0.001,      # Ultra-high precision: 1mm
        rot_tol=0.003       # Ultra-high precision: 0.17 degrees
    )


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
    
    # Define realistic joint space problem (within joint limits)
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([0.8, 0.5, -0.3, 0.2, 0.6, -0.1])  # More conservative joint angles
    
    print(f"Start configuration: {start_config}")
    print(f"Goal configuration:  {goal_config}")
    
    # Plan with joint space AORRTC
    print("\nPlanning with joint space AORRTC...")
    import time
    start_time = time.time()
    joint_path = motion_planner.plan_joint_path(start_config, goal_config)
    planning_time = time.time() - start_time
    
    if joint_path:
        print(f"✅ Joint space path found in {planning_time:.2f}s with {len(joint_path)} waypoints")
        
        # Compute forward kinematics for visualization (subsample to avoid too many calculations)
        cartesian_waypoints = []
        sample_indices = list(range(0, len(joint_path), max(1, len(joint_path) // 10)))  # Sample max 10 points
        
        for i in sample_indices:
            config = joint_path[i]
            try:
                T = robot_controller.forward_kinematics(config)
                cartesian_waypoints.append(T[:3, 3])
            except Exception as e:
                print(f"FK failed for config {i}: {e}")
        
        if cartesian_waypoints:
            print(f"Computed {len(cartesian_waypoints)} Cartesian waypoints for visualization")
            
            # Create simple visualization of end-effector path
            try:
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
            except Exception as e:
                print(f"❌ Visualization failed: {e}")
        
        # Compute path statistics
        path_length = sum(
            np.linalg.norm(joint_path[i+1] - joint_path[i]) 
            for i in range(len(joint_path)-1)
        )
        print(f"Joint space path length: {path_length:.3f} rad")
        
        # Validate configurations (sample a few)
        print("Validating sample joint configurations...")
        valid_count = 0
        sample_count = min(5, len(joint_path))
        for i in range(0, len(joint_path), len(joint_path) // sample_count):
            config = joint_path[i]
            try:
                T = robot_controller.forward_kinematics(config)
                valid_count += 1
            except Exception as e:
                print(f"❌ Configuration {i} invalid: {e}")
        
        print(f"✅ {valid_count}/{sample_count} sampled configurations validated successfully")
        
    else:
        print(f"❌ No joint space path found (took {planning_time:.2f}s)")


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

