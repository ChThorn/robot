"""
Visualization tools for robot planning using Plotly and Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Optional, Tuple, Dict, Any
import logging
import os
import sys

# Import robot kinematics for visualization
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))
try:
    from robot_controller import RobotController
    from robot_kinematics import RobotKinematics
except ImportError as e:
    logging.warning(f"Robot kinematics not available for visualization: {e}")
    RobotController = None
    RobotKinematics = None

from .trajectory_planning import TrajectoryInterpolator
from .path_planning import Environment3D

logger = logging.getLogger(__name__)

# Type aliases
JointPath = List[np.ndarray]
CartesianPath = List[np.ndarray]


class PlanningVisualizer:
    """Main visualization class for robot planning."""
    
    def __init__(self, robot_controller: Optional[RobotController] = None):
        """
        Initialize visualizer.
        
        Args:
            robot_controller: Optional robot controller for FK visualization
        """
        self.robot_controller = robot_controller
        self.colors = {
            'path': 'blue',
            'start': 'green', 
            'goal': 'red',
            'obstacle': 'gray',
            'tree_a': 'lightblue',
            'tree_b': 'orange',
            'trajectory': 'purple'
        }
    
    def plot_joint_trajectory(self, trajectory: TrajectoryInterpolator, 
                             save_path: Optional[str] = None,
                             show_velocity: bool = True,
                             show_acceleration: bool = True) -> go.Figure:
        """
        Plot joint space trajectory with position, velocity, and acceleration.
        
        Args:
            trajectory: Trajectory interpolator
            save_path: Optional path to save the plot
            show_velocity: Whether to show velocity plots
            show_acceleration: Whether to show acceleration plots
            
        Returns:
            Plotly figure
        """
        duration = trajectory.get_duration()
        times = np.linspace(0, duration, 200)
        
        # Sample trajectory
        positions = np.array([trajectory.evaluate(t) for t in times])
        
        if show_velocity:
            velocities = np.array([trajectory.evaluate_velocity(t) for t in times])
        if show_acceleration:
            accelerations = np.array([trajectory.evaluate_acceleration(t) for t in times])
        
        # Create subplots
        subplot_count = 1 + int(show_velocity) + int(show_acceleration)
        subplot_titles = ['Joint Positions']
        if show_velocity:
            subplot_titles.append('Joint Velocities')
        if show_acceleration:
            subplot_titles.append('Joint Accelerations')
        
        fig = make_subplots(
            rows=subplot_count, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08
        )
        
        joint_names = [f'Joint {i+1}' for i in range(positions.shape[1])]
        colors = px.colors.qualitative.Set1[:positions.shape[1]]
        
        # Plot positions
        for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
            fig.add_trace(
                go.Scatter(x=times, y=positions[:, i], 
                          name=f'{joint_name} Pos', 
                          line=dict(color=color)),
                row=1, col=1
            )
        
        # Plot velocities
        if show_velocity:
            for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
                fig.add_trace(
                    go.Scatter(x=times, y=velocities[:, i], 
                              name=f'{joint_name} Vel',
                              line=dict(color=color, dash='dash'),
                              showlegend=False),
                    row=2, col=1
                )
        
        # Plot accelerations
        if show_acceleration:
            row_idx = 2 + int(show_velocity)
            for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
                fig.add_trace(
                    go.Scatter(x=times, y=accelerations[:, i], 
                              name=f'{joint_name} Acc',
                              line=dict(color=color, dash='dot'),
                              showlegend=False),
                    row=row_idx, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Joint Space Trajectory',
            height=300 * subplot_count,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Time (s)')
        fig.update_yaxes(title_text='Position (rad)', row=1, col=1)
        if show_velocity:
            fig.update_yaxes(title_text='Velocity (rad/s)', row=2, col=1)
        if show_acceleration:
            row_idx = 2 + int(show_velocity)
            fig.update_yaxes(title_text='Acceleration (rad/s²)', row=row_idx, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Joint trajectory plot saved to {save_path}")
        
        return fig
    
    def plot_cartesian_path(self, path: CartesianPath, 
                           environment: Optional[Environment3D] = None,
                           save_path: Optional[str] = None,
                           show_orientation: bool = False) -> go.Figure:
        """
        Plot 3D Cartesian path with optional environment obstacles.
        
        Args:
            path: List of 4x4 transformation matrices or 3D positions
            environment: Optional environment with obstacles
            save_path: Optional path to save the plot
            show_orientation: Whether to show orientation frames
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Extract positions
        if len(path) > 0:
            if path[0].shape == (4, 4):
                # Transformation matrices
                positions = np.array([T[:3, 3] for T in path])
                orientations = [T[:3, :3] for T in path] if show_orientation else None
            else:
                # Just positions
                positions = np.array(path)
                orientations = None
        else:
            positions = np.array([])
            orientations = None
        
        # Plot path
        if len(positions) > 0:
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode='lines+markers',
                line=dict(color=self.colors['path'], width=4),
                marker=dict(size=3),
                name='Path'
            ))
            
            # Mark start and goal
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                mode='markers',
                marker=dict(size=10, color=self.colors['start'], symbol='circle'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                mode='markers',
                marker=dict(size=10, color=self.colors['goal'], symbol='square'),
                name='Goal'
            ))
            
            # Show orientation frames
            if show_orientation and orientations:
                self._add_orientation_frames(fig, positions[::5], orientations[::5])
        
        # Plot environment obstacles
        if environment:
            self._add_environment_obstacles(fig, environment)
        
        # Update layout
        fig.update_layout(
            title='Cartesian Path Visualization',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)', 
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cartesian path plot saved to {save_path}")
        
        return fig
    
    def plot_planning_tree(self, tree_start: Dict, tree_goal: Dict,
                          path: Optional[List[np.ndarray]] = None,
                          environment: Optional[Environment3D] = None,
                          save_path: Optional[str] = None) -> go.Figure:
        """
        Plot AORRTC planning trees with optional solution path.
        
        Args:
            tree_start: Start tree data structure
            tree_goal: Goal tree data structure  
            path: Optional solution path
            environment: Optional environment
            save_path: Optional save path
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Plot tree edges
        self._plot_tree_edges(fig, tree_start, self.colors['tree_a'], 'Start Tree')
        self._plot_tree_edges(fig, tree_goal, self.colors['tree_b'], 'Goal Tree')
        
        # Plot solution path
        if path:
            path_array = np.array(path)
            if path_array.shape[1] == 3:  # 3D positions
                fig.add_trace(go.Scatter3d(
                    x=path_array[:, 0], y=path_array[:, 1], z=path_array[:, 2],
                    mode='lines+markers',
                    line=dict(color=self.colors['path'], width=6),
                    marker=dict(size=4),
                    name='Solution Path'
                ))
        
        # Plot environment
        if environment:
            self._add_environment_obstacles(fig, environment)
        
        # Update layout
        fig.update_layout(
            title='AORRTC Planning Trees',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Planning tree plot saved to {save_path}")
        
        return fig
    
    def animate_trajectory_execution(self, trajectory: TrajectoryInterpolator,
                                   save_path: Optional[str] = None,
                                   fps: int = 30) -> go.Figure:
        """
        Create animated visualization of trajectory execution.
        
        Args:
            trajectory: Trajectory to animate
            save_path: Optional path to save animation
            fps: Frames per second
            
        Returns:
            Plotly figure with animation
        """
        if not self.robot_controller:
            logger.warning("Robot controller required for trajectory animation")
            return go.Figure()
        
        duration = trajectory.get_duration()
        dt = 1.0 / fps
        times = np.arange(0, duration + dt, dt)
        
        # Compute forward kinematics for each time step
        positions = []
        for t in times:
            q = trajectory.evaluate(t)
            try:
                T = self.robot_controller.forward_kinematics(q)
                positions.append(T[:3, 3])
            except Exception as e:
                logger.warning(f"FK failed at t={t}: {e}")
                positions.append(positions[-1] if positions else np.zeros(3))
        
        positions = np.array(positions)
        
        # Create animated scatter plot
        fig = go.Figure()
        
        # Add trajectory trace
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='lines',
            line=dict(color=self.colors['trajectory'], width=2),
            name='Trajectory',
            opacity=0.3
        ))
        
        # Add animated end-effector position
        frames = []
        for i, pos in enumerate(positions):
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode='markers',
                        marker=dict(size=8, color=self.colors['goal']),
                        name='End-Effector'
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
            title='Trajectory Execution Animation',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': int(1000/fps)}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Trajectory animation saved to {save_path}")
        
        return fig
    
    def _plot_tree_edges(self, fig: go.Figure, tree: Dict, color: str, name: str):
        """Add tree edges to plotly figure."""
        if 'points' not in tree or 'parents' not in tree:
            return
        
        x_coords, y_coords, z_coords = [], [], []
        
        for i, parent_idx in enumerate(tree['parents']):
            if parent_idx != -1:
                p1, p2 = tree['points'][parent_idx], tree['points'][i]
                
                # Handle different point formats
                if len(p1) >= 3 and len(p2) >= 3:
                    x_coords.extend([p1[0], p2[0], None])
                    y_coords.extend([p1[1], p2[1], None])
                    z_coords.extend([p1[2], p2[2], None])
        
        if x_coords:
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color=color, width=1),
                name=name,
                hoverinfo='skip'
            ))
    
    def _add_environment_obstacles(self, fig: go.Figure, environment: Environment3D):
        """Add environment obstacles to plotly figure."""
        # Add sphere obstacles
        for i, (center, radius) in enumerate(environment.sphere_obstacles):
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:15j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, self.colors['obstacle']], [1, self.colors['obstacle']]],
                opacity=0.7,
                showscale=False,
                name=f'Sphere Obstacle {i+1}'
            ))
        
        # Add box obstacles
        for i, (min_corner, max_corner) in enumerate(environment.box_obstacles):
            # Create box wireframe
            vertices = [
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]]
            ]
            
            # Define box edges
            edges = [
                [0,1], [1,2], [2,3], [3,0],  # bottom face
                [4,5], [5,6], [6,7], [7,4],  # top face
                [0,4], [1,5], [2,6], [3,7]   # vertical edges
            ]
            
            x_coords, y_coords, z_coords = [], [], []
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                x_coords.extend([v1[0], v2[0], None])
                y_coords.extend([v1[1], v2[1], None])
                z_coords.extend([v1[2], v2[2], None])
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color=self.colors['obstacle'], width=3),
                name=f'Box Obstacle {i+1}',
                hoverinfo='skip'
            ))
    
    def _add_orientation_frames(self, fig: go.Figure, positions: np.ndarray, 
                               orientations: List[np.ndarray], scale: float = 0.05):
        """Add orientation frames to plotly figure."""
        colors = ['red', 'green', 'blue']  # X, Y, Z axes
        
        for pos, R in zip(positions, orientations):
            for i, color in enumerate(colors):
                axis = R[:, i] * scale
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + axis[0]],
                    y=[pos[1], pos[1] + axis[1]],
                    z=[pos[2], pos[2] + axis[2]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))


def create_planning_dashboard(trajectory: TrajectoryInterpolator,
                             cartesian_path: Optional[CartesianPath] = None,
                             environment: Optional[Environment3D] = None,
                             robot_controller: Optional[RobotController] = None,
                             save_path: Optional[str] = None) -> go.Figure:
    """
    Create comprehensive planning dashboard with multiple visualizations.
    
    Args:
        trajectory: Trajectory to visualize
        cartesian_path: Optional Cartesian path
        environment: Optional environment
        robot_controller: Optional robot controller
        save_path: Optional save path
        
    Returns:
        Plotly dashboard figure
    """
    visualizer = PlanningVisualizer(robot_controller)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Joint Trajectory', 'Cartesian Path', 'Joint Velocities', 'Joint Accelerations'],
        specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Get trajectory data
    duration = trajectory.get_duration()
    times = np.linspace(0, duration, 200)
    positions = np.array([trajectory.evaluate(t) for t in times])
    velocities = np.array([trajectory.evaluate_velocity(t) for t in times])
    
    # Plot joint positions
    joint_names = [f'Joint {i+1}' for i in range(positions.shape[1])]
    colors = px.colors.qualitative.Set1[:positions.shape[1]]
    
    for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
        fig.add_trace(
            go.Scatter(x=times, y=positions[:, i], 
                      name=f'{joint_name}', 
                      line=dict(color=color)),
            row=1, col=1
        )
    
    # Plot Cartesian path if available
    if cartesian_path and robot_controller:
        cart_positions = []
        for q in [trajectory.evaluate(t) for t in times[::10]]:  # Subsample for performance
            try:
                T = robot_controller.forward_kinematics(q)
                cart_positions.append(T[:3, 3])
            except:
                pass
        
        if cart_positions:
            cart_positions = np.array(cart_positions)
            fig.add_trace(
                go.Scatter3d(x=cart_positions[:, 0], y=cart_positions[:, 1], z=cart_positions[:, 2],
                            mode='lines+markers',
                            line=dict(color='blue', width=4),
                            name='End-Effector Path'),
                row=1, col=2
            )
    
    # Plot joint velocities
    for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
        fig.add_trace(
            go.Scatter(x=times, y=velocities[:, i], 
                      name=f'{joint_name} Vel',
                      line=dict(color=color, dash='dash'),
                      showlegend=False),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Robot Planning Dashboard',
        height=800,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Planning dashboard saved to {save_path}")
    
    return fig

