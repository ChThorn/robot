"""
Utility functions and configuration management for robot planning.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlanningConfig:
    """Configuration class for planning parameters."""
    
    # AORRTC parameters
    max_iterations: int = 5000
    step_size: float = 0.35
    goal_bias: float = 0.3
    connect_threshold: float = 0.45
    rewire_radius: float = 0.8
    line_bias: float = 0.4
    patience_no_improve: int = 800
    max_time: float = 3.0
    
    # Trajectory parameters
    max_joint_velocities: np.ndarray = field(default_factory=lambda: np.full(6, 3.0))
    max_joint_accelerations: np.ndarray = field(default_factory=lambda: np.full(6, 10.0))
    max_cartesian_velocity: float = 1.0
    max_cartesian_acceleration: float = 5.0
    
    # Visualization parameters
    show_trees: bool = True
    show_path: bool = True
    show_obstacles: bool = True
    animation_speed: float = 0.1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PlanningConfig':
        """Create config from dictionary."""
        # Handle numpy arrays
        if 'max_joint_velocities' in config_dict:
            config_dict['max_joint_velocities'] = np.array(config_dict['max_joint_velocities'])
        if 'max_joint_accelerations' in config_dict:
            config_dict['max_joint_accelerations'] = np.array(config_dict['max_joint_accelerations'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            else:
                config_dict[key] = value
        return config_dict


def load_config(config_path: Optional[str] = None) -> PlanningConfig:
    """
    Load planning configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
        
    Returns:
        PlanningConfig object
    """
    if config_path is None:
        # Use default config file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return PlanningConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    logger.info("Using default configuration")
    return PlanningConfig()


def save_config(config: PlanningConfig, config_path: str):
    """
    Save planning configuration to YAML file.
    
    Args:
        config: PlanningConfig object to save
        config_path: Path where to save the configuration
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")


def validate_joint_configuration(q: np.ndarray, joint_limits: np.ndarray, 
                                margin: float = 0.01) -> bool:
    """
    Validate joint configuration against limits.
    
    Args:
        q: Joint configuration
        joint_limits: Joint limits [lower, upper]
        margin: Safety margin (fraction of range)
        
    Returns:
        True if configuration is valid
    """
    if not isinstance(q, np.ndarray) or q.ndim != 1:
        return False
    
    # Check if dimensions match
    if len(q) != joint_limits.shape[1]:
        return False
    
    limits_lower, limits_upper = joint_limits[0], joint_limits[1]
    limit_range = limits_upper - limits_lower
    margin_abs = limit_range * margin
    
    effective_lower = limits_lower + margin_abs
    effective_upper = limits_upper - margin_abs
    
    return np.all(q >= effective_lower) and np.all(q <= effective_upper)


def interpolate_joint_path(start: np.ndarray, goal: np.ndarray, 
                          num_points: int = 50) -> np.ndarray:
    """
    Interpolate between two joint configurations.
    
    Args:
        start: Start joint configuration
        goal: Goal joint configuration
        num_points: Number of interpolation points
        
    Returns:
        Array of interpolated joint configurations
    """
    return np.linspace(start, goal, num_points)


def compute_path_length(path: np.ndarray) -> float:
    """
    Compute total length of a joint space path.
    
    Args:
        path: Array of joint configurations [n_points, n_joints]
        
    Returns:
        Total path length
    """
    if len(path) < 2:
        return 0.0
    
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def smooth_joint_path(path: np.ndarray, smoothing_factor: float = 0.1, 
                     iterations: int = 10) -> np.ndarray:
    """
    Apply simple smoothing to a joint space path.
    
    Args:
        path: Array of joint configurations [n_points, n_joints]
        smoothing_factor: Smoothing strength (0-1)
        iterations: Number of smoothing iterations
        
    Returns:
        Smoothed path
    """
    if len(path) < 3:
        return path.copy()
    
    smoothed = path.copy()
    
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):
            # Simple averaging with neighbors
            neighbor_avg = (smoothed[i-1] + smoothed[i+1]) / 2
            smoothed[i] = (1 - smoothing_factor) * smoothed[i] + smoothing_factor * neighbor_avg
    
    return smoothed

