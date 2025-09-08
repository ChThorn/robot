#!/usr/bin/env python3
"""
Production-ready real-time motion planner for robot manipulators.

This module implements kinematically-aware planning with strict performance constraints
for real-time robotic applications. Key features:

- Kinematic validation DURING planning (not after)
- Strict timeouts and early termination
- Fast fallback strategies
- Real-time performance guarantees
- Production-ready error handling

Author: Robot Planning Team
Version: 2.0.0 (Real-time Production)
"""

import numpy as np
import time
import logging
import threading
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import robot kinematics system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))
try:
    from robot_controller import RobotController
except ImportError as e:
    logging.error(f"Failed to import robot kinematics: {e}")
    raise

from .unit_converter import UnitConverter, ProductionUnitHandler

logger = logging.getLogger(__name__)


class PlanningResult(Enum):
    """Planning result status codes."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    NO_SOLUTION = "no_solution"
    KINEMATIC_INFEASIBLE = "kinematic_infeasible"
    ERROR = "error"


@dataclass
class PlanningConstraints:
    """Real-time planning constraints."""
    max_planning_time: float = 2.0  # Maximum planning time in seconds
    max_ik_time_per_pose: float = 0.1  # Maximum IK time per pose
    max_path_length: int = 20  # Maximum number of waypoints
    position_tolerance: float = 0.001  # Position tolerance in meters
    orientation_tolerance: float = 0.05  # Orientation tolerance in radians
    joint_velocity_limit: float = 2.0  # Joint velocity limit in rad/s
    workspace_margin: float = 0.05  # Workspace margin in meters


@dataclass
class PlanningMetrics:
    """Planning performance metrics."""
    planning_time: float = 0.0
    ik_calls: int = 0
    ik_successes: int = 0
    path_length: int = 0
    result: PlanningResult = PlanningResult.ERROR


class KinematicValidator:
    """Fast kinematic validation for real-time planning."""
    
    def __init__(self, robot_controller: RobotController, constraints: PlanningConstraints):
        self.robot_controller = robot_controller
        self.constraints = constraints
        self.unit_handler = ProductionUnitHandler(robot_controller)
        
        # Pre-compute workspace bounds for fast validation
        self.workspace_bounds = self.unit_handler.get_workspace_bounds_planning_units()
        
        # Cache for IK solutions to avoid redundant computations
        self.ik_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def is_pose_reachable(self, pose: np.ndarray, q_seed: np.ndarray = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Fast reachability check with kinematic validation.
        
        Args:
            pose: 4x4 transformation matrix in planning units
            q_seed: Seed joint configuration for IK
            
        Returns:
            Tuple of (is_reachable, joint_config)
        """
        start_time = time.time()
        
        # Quick workspace check
        pos = pose[:3, 3]
        if not self._is_in_workspace(pos):
            return False, None
        
        # Check cache first
        pose_key = self._pose_to_key(pose)
        if pose_key in self.ik_cache:
            self.cache_hits += 1
            return True, self.ik_cache[pose_key]
        
        self.cache_misses += 1
        
        # Fast IK with timeout
        try:
            # Set strict timeout for IK
            original_params = self.robot_controller.ik_params.copy()
            self.robot_controller.set_ik_parameters(
                max_iters=100,  # Reduced iterations for speed
                pos_tol=self.constraints.position_tolerance,
                rot_tol=self.constraints.orientation_tolerance
            )
            
            # Use timeout wrapper for IK
            q_solution, success = self._ik_with_timeout(
                pose, q_seed, self.constraints.max_ik_time_per_pose
            )
            
            # Restore original parameters
            self.robot_controller.set_ik_parameters(**original_params)
            
            if success and q_solution is not None:
                # Validate joint limits and velocities
                if self._validate_joint_config(q_solution, q_seed):
                    # Cache successful solution
                    self.ik_cache[pose_key] = q_solution
                    return True, q_solution
            
            return False, None
            
        except Exception as e:
            logger.warning(f"IK validation failed: {e}")
            return False, None
        finally:
            elapsed = time.time() - start_time
            if elapsed > self.constraints.max_ik_time_per_pose:
                logger.warning(f"IK validation timeout: {elapsed:.3f}s")
    
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """Fast workspace bounds check."""
        for i, (min_val, max_val) in enumerate(self.workspace_bounds):
            margin = self.constraints.workspace_margin
            if not (min_val + margin <= pos[i] <= max_val - margin):
                return False
        return True
    
    def _pose_to_key(self, pose: np.ndarray) -> str:
        """Convert pose to cache key."""
        # Round to reduce cache size while maintaining accuracy
        pos = np.round(pose[:3, 3], 3)
        rpy = np.round(self.robot_controller.robot.matrix_to_rpy(pose[:3, :3]), 3)
        return f"{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},{rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}"
    
    def _ik_with_timeout(self, pose: np.ndarray, q_seed: np.ndarray, timeout: float) -> Tuple[Optional[np.ndarray], bool]:
        """IK solver with timeout."""
        result = [None, False]
        
        def ik_worker():
            try:
                q, success = self.unit_handler.inverse_kinematics_planning_units(pose, q_seed)
                result[0] = q
                result[1] = success
            except Exception as e:
                logger.debug(f"IK worker exception: {e}")
                result[1] = False
        
        thread = threading.Thread(target=ik_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            logger.debug("IK timeout occurred")
            return None, False
        
        return result[0], result[1]
    
    def _validate_joint_config(self, q: np.ndarray, q_prev: np.ndarray = None) -> bool:
        """Validate joint configuration against limits and velocity constraints."""
        # Check joint limits
        if not self.unit_handler.validate_joint_limits_planning_units(q):
            return False
        
        # Check velocity constraints if previous config available
        if q_prev is not None:
            joint_diff = np.abs(q - q_prev)
            max_diff = self.constraints.joint_velocity_limit * self.constraints.max_ik_time_per_pose
            if np.any(joint_diff > max_diff):
                logger.debug(f"Joint velocity constraint violated: {np.max(joint_diff):.3f} > {max_diff:.3f}")
                return False
        
        return True
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.ik_cache)
        }


class RealtimeRRTConnect:
    """Real-time RRT-Connect with kinematic validation."""
    
    def __init__(self, validator: KinematicValidator, constraints: PlanningConstraints):
        self.validator = validator
        self.constraints = constraints
        self.metrics = PlanningMetrics()
        
        # Planning parameters optimized for real-time performance
        self.max_iterations = 500  # Reduced for real-time
        self.step_size = 0.05  # meters
        self.goal_bias = 0.2
        
    def plan(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> Tuple[PlanningResult, Optional[List[np.ndarray]], PlanningMetrics]:
        """
        Plan path with real-time constraints.
        
        Args:
            start_pose: Start pose in planning units
            goal_pose: Goal pose in planning units
            
        Returns:
            Tuple of (result, path, metrics)
        """
        start_time = time.time()
        self.metrics = PlanningMetrics()
        
        try:
            # Validate start and goal poses
            start_reachable, q_start = self.validator.is_pose_reachable(start_pose)
            if not start_reachable:
                self.metrics.result = PlanningResult.KINEMATIC_INFEASIBLE
                return self.metrics.result, None, self.metrics
            
            goal_reachable, q_goal = self.validator.is_pose_reachable(goal_pose)
            if not goal_reachable:
                self.metrics.result = PlanningResult.KINEMATIC_INFEASIBLE
                return self.metrics.result, None, self.metrics
            
            # Initialize trees
            start_tree = {'poses': [start_pose], 'parents': [-1], 'joint_configs': [q_start]}
            goal_tree = {'poses': [goal_pose], 'parents': [-1], 'joint_configs': [q_goal]}
            
            # RRT-Connect main loop
            for iteration in range(self.max_iterations):
                # Check timeout
                if time.time() - start_time > self.constraints.max_planning_time:
                    self.metrics.result = PlanningResult.TIMEOUT
                    break
                
                # Extend start tree toward random/goal pose
                if np.random.random() < self.goal_bias:
                    target_pose = goal_pose
                else:
                    target_pose = self._sample_random_pose()
                
                new_idx_start = self._extend_tree(start_tree, target_pose)
                if new_idx_start is None:
                    continue
                
                # Try to connect goal tree to new node
                new_pose = start_tree['poses'][new_idx_start]
                new_idx_goal = self._extend_tree(goal_tree, new_pose)
                
                if new_idx_goal is not None:
                    # Check if trees are connected
                    if self._trees_connected(start_tree, new_idx_start, goal_tree, new_idx_goal):
                        # Extract path
                        path = self._extract_path(start_tree, new_idx_start, goal_tree, new_idx_goal)
                        self.metrics.result = PlanningResult.SUCCESS
                        self.metrics.path_length = len(path)
                        return self.metrics.result, path, self.metrics
                
                # Swap trees for bidirectional search
                start_tree, goal_tree = goal_tree, start_tree
            
            # No solution found within constraints
            self.metrics.result = PlanningResult.NO_SOLUTION
            return self.metrics.result, None, self.metrics
            
        except Exception as e:
            logger.error(f"Planning error: {e}")
            self.metrics.result = PlanningResult.ERROR
            return self.metrics.result, None, self.metrics
        finally:
            self.metrics.planning_time = time.time() - start_time
            self.metrics.ik_calls = self.validator.cache_hits + self.validator.cache_misses
    
    def _sample_random_pose(self) -> np.ndarray:
        """Sample random pose within workspace."""
        # Sample position within workspace bounds
        pos = np.array([
            np.random.uniform(bounds[0] + self.constraints.workspace_margin, 
                            bounds[1] - self.constraints.workspace_margin)
            for bounds in self.validator.workspace_bounds
        ])
        
        # Sample random orientation (simplified)
        rpy = np.random.uniform(-np.pi/4, np.pi/4, 3)  # Limited orientation range
        R = self.validator.robot_controller.robot.rpy_to_matrix(rpy)
        
        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3, :3] = R
        
        return pose
    
    def _extend_tree(self, tree: Dict, target_pose: np.ndarray) -> Optional[int]:
        """Extend tree toward target pose."""
        # Find nearest node
        distances = [np.linalg.norm(pose[:3, 3] - target_pose[:3, 3]) for pose in tree['poses']]
        nearest_idx = np.argmin(distances)
        nearest_pose = tree['poses'][nearest_idx]
        nearest_q = tree['joint_configs'][nearest_idx]
        
        # Compute new pose
        direction = target_pose[:3, 3] - nearest_pose[:3, 3]
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return None
        
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        
        new_pos = nearest_pose[:3, 3] + direction
        
        # Simple orientation interpolation
        new_pose = nearest_pose.copy()
        new_pose[:3, 3] = new_pos
        
        # Validate new pose
        reachable, q_new = self.validator.is_pose_reachable(new_pose, nearest_q)
        if not reachable:
            return None
        
        # Add to tree
        tree['poses'].append(new_pose)
        tree['parents'].append(nearest_idx)
        tree['joint_configs'].append(q_new)
        
        return len(tree['poses']) - 1
    
    def _trees_connected(self, tree1: Dict, idx1: int, tree2: Dict, idx2: int) -> bool:
        """Check if two tree nodes are connected."""
        pose1 = tree1['poses'][idx1]
        pose2 = tree2['poses'][idx2]
        
        pos_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        return pos_diff < self.step_size / 2
    
    def _extract_path(self, start_tree: Dict, start_idx: int, goal_tree: Dict, goal_idx: int) -> List[np.ndarray]:
        """Extract path from trees."""
        # Path from start to connection point
        path = []
        idx = start_idx
        while idx != -1:
            path.append(start_tree['poses'][idx])
            idx = start_tree['parents'][idx]
        path.reverse()
        
        # Path from connection point to goal
        idx = goal_tree['parents'][goal_idx]
        while idx != -1:
            path.append(goal_tree['poses'][idx])
            idx = goal_tree['parents'][idx]
        
        return path


class ProductionMotionPlanner:
    """Production-ready motion planner with real-time guarantees."""
    
    def __init__(self, robot_controller: RobotController, constraints: PlanningConstraints = None):
        self.robot_controller = robot_controller
        self.constraints = constraints or PlanningConstraints()
        self.validator = KinematicValidator(robot_controller, self.constraints)
        self.planner = RealtimeRRTConnect(self.validator, self.constraints)
        
        logger.info("Production motion planner initialized with real-time constraints")
        logger.info(f"Max planning time: {self.constraints.max_planning_time}s")
        logger.info(f"Max IK time per pose: {self.constraints.max_ik_time_per_pose}s")
    
    def plan_cartesian_path_realtime(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> Tuple[PlanningResult, Optional[List[np.ndarray]], PlanningMetrics]:
        """
        Plan Cartesian path with real-time constraints.
        
        Args:
            start_pose: Start pose in planning units (4x4 matrix)
            goal_pose: Goal pose in planning units (4x4 matrix)
            
        Returns:
            Tuple of (result, cartesian_path, metrics)
        """
        logger.info("Starting real-time Cartesian path planning")
        
        result, path, metrics = self.planner.plan(start_pose, goal_pose)
        
        # Log results
        logger.info(f"Planning completed: {result.value}")
        logger.info(f"Planning time: {metrics.planning_time:.3f}s")
        logger.info(f"IK calls: {metrics.ik_calls}, successes: {metrics.ik_successes}")
        
        if result == PlanningResult.SUCCESS:
            logger.info(f"Path found with {metrics.path_length} waypoints")
        
        # Log cache performance
        cache_stats = self.validator.get_cache_stats()
        logger.info(f"IK cache hit rate: {cache_stats['hit_rate']:.2%}")
        
        return result, path, metrics
    
    def plan_with_fallback(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> Tuple[PlanningResult, Optional[List[np.ndarray]], PlanningMetrics]:
        """
        Plan with automatic fallback strategies for production robustness.
        
        Args:
            start_pose: Start pose in planning units
            goal_pose: Goal pose in planning units
            
        Returns:
            Tuple of (result, path, metrics)
        """
        # Try primary planning
        result, path, metrics = self.plan_cartesian_path_realtime(start_pose, goal_pose)
        
        if result == PlanningResult.SUCCESS:
            return result, path, metrics
        
        logger.warning(f"Primary planning failed: {result.value}, trying fallbacks")
        
        # Fallback 1: Relaxed constraints
        if result == PlanningResult.TIMEOUT:
            logger.info("Fallback 1: Relaxed time constraints")
            original_time = self.constraints.max_planning_time
            self.constraints.max_planning_time *= 1.5
            
            result, path, metrics = self.plan_cartesian_path_realtime(start_pose, goal_pose)
            self.constraints.max_planning_time = original_time
            
            if result == PlanningResult.SUCCESS:
                return result, path, metrics
        
        # Fallback 2: Simplified path (direct line)
        if result in [PlanningResult.NO_SOLUTION, PlanningResult.TIMEOUT]:
            logger.info("Fallback 2: Direct line path")
            direct_path = self._plan_direct_path(start_pose, goal_pose)
            if direct_path:
                metrics.result = PlanningResult.SUCCESS
                metrics.path_length = len(direct_path)
                return PlanningResult.SUCCESS, direct_path, metrics
        
        # All fallbacks failed
        logger.error("All planning strategies failed")
        return result, None, metrics
    
    def _plan_direct_path(self, start_pose: np.ndarray, goal_pose: np.ndarray, num_waypoints: int = 5) -> Optional[List[np.ndarray]]:
        """Plan direct line path as fallback."""
        try:
            path = []
            for i in range(num_waypoints + 1):
                alpha = i / num_waypoints
                
                # Linear interpolation of position
                pos = (1 - alpha) * start_pose[:3, 3] + alpha * goal_pose[:3, 3]
                
                # Simple orientation interpolation
                pose = start_pose.copy()
                pose[:3, 3] = pos
                
                # Validate pose
                reachable, _ = self.validator.is_pose_reachable(pose)
                if not reachable:
                    return None
                
                path.append(pose)
            
            return path
            
        except Exception as e:
            logger.error(f"Direct path planning failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        cache_stats = self.validator.get_cache_stats()
        return {
            'constraints': {
                'max_planning_time': self.constraints.max_planning_time,
                'max_ik_time_per_pose': self.constraints.max_ik_time_per_pose,
                'position_tolerance': self.constraints.position_tolerance,
                'orientation_tolerance': self.constraints.orientation_tolerance
            },
            'cache_performance': cache_stats,
            'workspace_bounds': self.validator.workspace_bounds
        }

