#!/usr/bin/env python3
"""
Motion planning integration layer between path planning and robot kinematics.
Handles joint space planning and robot-specific constraints.
Production-ready with proper unit conversions between planning (SI) and robot (mm/degrees) units.
"""

import numpy as np
import sys
import os
import logging
import time
from typing import List, Tuple, Optional, Dict, Union
from scipy.spatial import cKDTree

# Import robot kinematics system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))
try:
    from robot_kinematics import RobotKinematics, RobotKinematicsError
    from robot_controller import RobotController
except ImportError as e:
    logging.error(f"Failed to import robot kinematics: {e}")
    logging.error("Ensure robot_kinematics folder is accessible")
    raise

from .path_planning import CartesianPathPlanner, Environment3D
from .unit_converter import UnitConverter, ProductionUnitHandler

logger = logging.getLogger(__name__)

# Type aliases
JointConfiguration = np.ndarray
CartesianPose = np.ndarray
JointPath = List[JointConfiguration]
CartesianPath = List[CartesianPose]


class RobotEnvironment(Environment3D):
    """Extended environment with robot-specific constraints."""
    
    def __init__(self, robot_controller: RobotController):
        """
        Initialize robot environment from robot workspace constraints.
        
        Args:
            robot_controller: Robot controller with kinematics and constraints
        """
        self.robot_controller = robot_controller
        self.robot = robot_controller.robot
        
        # Extract workspace bounds from robot constraints
        workspace_bounds = self._extract_workspace_bounds()
        super().__init__(workspace_bounds)
        
        # Load obstacles from robot constraints
        self._load_robot_obstacles()
        
    def _extract_workspace_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Extract workspace bounds from robot constraints."""
        constraints = self.robot.constraints
        
        if 'workspace' in constraints:
            ws = constraints['workspace']
            # Convert from mm to meters
            x_bounds = (ws['x_min'] / 1000.0, ws['x_max'] / 1000.0)
            y_bounds = (ws['y_min'] / 1000.0, ws['y_max'] / 1000.0)
            z_bounds = (ws['z_min'] / 1000.0, ws['z_max'] / 1000.0)
            return (x_bounds, y_bounds, z_bounds)
        else:
            # Default workspace bounds if not specified
            logger.warning("No workspace constraints found, using default bounds")
            return ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
            
    def _load_robot_obstacles(self):
        """Load obstacles from robot constraints file."""
        constraints = self.robot.constraints
        
        if 'obstacles' in constraints:
            for obstacle in constraints['obstacles']:
                if obstacle['type'] == 'box':
                    center = np.array(obstacle['center']) / 1000.0  # Convert mm to m
                    size = np.array(obstacle['size']) / 1000.0
                    min_corner = center - size / 2
                    max_corner = center + size / 2
                    self.add_box_obstacle(min_corner.tolist(), max_corner.tolist())
                    logger.debug(f"Added box obstacle: center={center}, size={size}")
                    
                elif obstacle['type'] == 'cylinder':
                    center = np.array(obstacle['center']) / 1000.0  # Convert mm to m
                    radius = obstacle['radius'] / 1000.0
                    height = obstacle['height'] / 1000.0
                    self.add_cylinder_obstacle(center.tolist(), radius, height)
                    logger.debug(f"Added cylinder obstacle: center={center}, radius={radius}, height={height}")
                    
        logger.info(f"Loaded {len(self.sphere_obstacles)} sphere, {len(self.box_obstacles)} box, "
                   f"and {len(self.cylinder_obstacles)} cylinder obstacles")
                    
    def is_pose_reachable(self, pose: CartesianPose) -> bool:
        """
        Check if Cartesian pose is reachable by robot.
        
        Args:
            pose: 4x4 transformation matrix or [x, y, z, rx, ry, rz]
            
        Returns:
            True if pose is reachable, False otherwise
        """
        if pose.shape == (4, 4):
            # Transformation matrix
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
        elif pose.shape == (6,):
            # Position and RPY
            position = pose[:3]
            rpy = pose[3:]
            rotation_matrix = self.robot.rpy_to_matrix(rpy)
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = position
        else:
            logger.error(f"Invalid pose format: {pose.shape}")
            return False
            
        # Check workspace bounds
        if not self.is_point_valid(position):
            return False
            
        # Check robot constraints
        if not self.robot._check_workspace(position):
            return False
            
        rpy = self.robot.matrix_to_rpy(rotation_matrix)
        if not self.robot._check_orientation(rpy):
            return False
            
        if not self.robot._check_obstacles(position):
            return False
            
        # Check if IK solution exists
        q_solution, converged = self.robot.inverse_kinematics(pose)
        return converged and q_solution is not None


class JointSpacePlanner:
    """AORRTC-based planner for joint space with goal/line bias and early exits."""

    def __init__(self, robot_controller: RobotController,
                 max_iterations: int = 2000,
                 step_size: float = 0.35,
                 goal_bias: float = 0.3,
                 connect_threshold: float = 0.45,
                 rewire_radius: float = 0.8,
                 line_bias: float = 0.4,
                 patience_no_improve: int = 800,
                 max_time: float = 3.0):
        self.robot_controller = robot_controller
        self.robot = robot_controller.robot
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.connect_threshold = connect_threshold
        self.rewire_radius = rewire_radius
        self.line_bias = line_bias
        self.patience_no_improve = patience_no_improve
        self.max_time = max_time

        # Joint limits
        self.joint_limits_lower = self.robot.joint_limits[0]
        self.joint_limits_upper = self.robot.joint_limits[1]

        # Planning state
        self.tree_start: Optional[Dict] = None
        self.tree_goal: Optional[Dict] = None
        self.best_path: Optional[JointPath] = None
        self.best_cost = float('inf')
        self._baseline_cost = None
        self._no_improve_iters = 0
        self._plan_start_time = None
        self._last_reported_cost = float('inf')
        self._start_config = None
        self._goal_config = None

    def plan(self, start_config: JointConfiguration, goal_config: JointConfiguration) -> Optional[JointPath]:
        start_config = np.array(start_config)
        goal_config = np.array(goal_config)

        # Validate configurations
        if not self._is_config_valid(start_config):
            logger.error("Start configuration is invalid")
            return None
        if not self._is_config_valid(goal_config):
            logger.error("Goal configuration is invalid")
            return None

        # Initialize trees
        self._initialize_planning(start_config, goal_config)
        self._plan_start_time = time.time()

        logger.info(f"Starting joint space planning (max_iter={self.max_iterations})")

        for i in range(self.max_iterations):
            # Time budget check
            if self.max_time is not None and (time.time() - self._plan_start_time) > self.max_time:
                if self.best_path:
                    logger.debug("Exiting due to time budget with a valid path available")
                    break

            # Progress logging
            if i % 1000 == 0 and i > 0:
                tree_sizes = f"{len(self.tree_start['configs'])}/{len(self.tree_goal['configs'])}"
                cost_info = f", Cost: {self.best_cost:.4f}" if self.best_path else ""
                logger.debug(f"Iteration {i}, Trees: {tree_sizes}{cost_info}")

            # Alternate between trees
            tree_from, tree_to = (self.tree_start, self.tree_goal) if i % 2 == 0 else (self.tree_goal, self.tree_start)

            # Sample and extend
            rand_config = self._sample_joint_space(tree_to)
            new_idx = self._extend_tree(tree_from, rand_config)

            if new_idx is not None:
                # Try to connect trees
                if self._try_connect(tree_from, tree_to, new_idx):
                    # Rewire for optimization
                    self._rewire_tree(tree_from, new_idx)
                    # Early-exit if we have a near-straight path cost
                    if self._baseline_cost is not None and self.best_cost <= 1.05 * self._baseline_cost:
                        logger.debug("Early exit: path cost near baseline straight-line")
                        break

            # Track improvement stall
            if self.best_path:
                if self.best_cost < self._last_reported_cost:
                    self._no_improve_iters = 0
                    self._last_reported_cost = self.best_cost
                else:
                    self._no_improve_iters += 1
                    if self._no_improve_iters >= self.patience_no_improve:
                        logger.debug("Early exit: no improvement for patience window")
                        break

        if self.best_path:
            logger.info(f"Joint space path found with cost: {self.best_cost:.4f}")
        else:
            logger.warning("No joint space path found")

        return self.best_path

    def _initialize_planning(self, start_config: JointConfiguration, goal_config: JointConfiguration):
        self.tree_start = {
            'configs': [start_config],
            'parents': [-1],
            'costs': [0.0]
        }
        self.tree_goal = {
            'configs': [goal_config],
            'parents': [-1],
            'costs': [0.0]
        }
        self.best_path = None
        self.best_cost = float('inf')
        self._baseline_cost = float(np.linalg.norm(goal_config - start_config))
        self._no_improve_iters = 0
        self._last_reported_cost = float('inf')
        self._start_config = start_config.copy()
        self._goal_config = goal_config.copy()

    def _sample_joint_space(self, tree_to: Dict) -> JointConfiguration:
        r = np.random.random()
        if r < self.goal_bias:
            return tree_to['configs'][0]
        elif r < self.goal_bias + self.line_bias:
            # Sample near straight line between start and goal
            alpha = np.random.random()
            base = (1 - alpha) * self._start_config + alpha * self._goal_config
            noise = np.random.normal(0, 0.15, size=base.shape)
            sample = np.clip(base + noise, self.joint_limits_lower, self.joint_limits_upper)
            return sample
        else:
            return np.random.uniform(self.joint_limits_lower, self.joint_limits_upper)

    def _extend_tree(self, tree: Dict, target_config: JointConfiguration) -> Optional[int]:
        if len(tree['configs']) == 0:
            return None

        # Find nearest neighbor
        configs_array = np.array(tree['configs'])
        distances = np.linalg.norm(configs_array - target_config, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_config = tree['configs'][nearest_idx]

        # Compute step toward target
        direction = target_config - nearest_config
        distance = np.linalg.norm(direction)

        if distance == 0:
            return None

        # Take step
        step_distance = min(self.step_size, distance)
        new_config = nearest_config + direction / distance * step_distance

        # Clamp to joint limits
        new_config = np.clip(new_config, self.joint_limits_lower, self.joint_limits_upper)

        # Validate configuration
        if not self._is_config_valid(new_config):
            return None

        # Check path validity (simplified collision checking)
        if not self._is_path_valid(nearest_config, new_config):
            return None

        # Add to tree
        new_cost = tree['costs'][nearest_idx] + step_distance
        tree['configs'].append(new_config)
        tree['parents'].append(nearest_idx)
        tree['costs'].append(new_cost)

        return len(tree['configs']) - 1

    def _try_connect(self, tree_from: Dict, tree_to: Dict, new_idx: int) -> bool:
        new_config = tree_from['configs'][new_idx]

        # Find nearest configuration in target tree
        configs_array = np.array(tree_to['configs'])
        distances = np.linalg.norm(configs_array - new_config, axis=1)
        nearest_idx_to = np.argmin(distances)
        distance = distances[nearest_idx_to]

        if distance < self.connect_threshold:
            nearest_config_to = tree_to['configs'][nearest_idx_to]
            if self._is_path_valid(new_config, nearest_config_to):
                self._update_best_path(tree_from, new_idx, tree_to, nearest_idx_to)
                return True

        return False

    def _update_best_path(self, tree1: Dict, idx1: int, tree2: Dict, idx2: int):
        # Extract paths from both trees
        path1 = self._extract_path(tree1, idx1)
        path2 = self._extract_path(tree2, idx2)

        # Combine paths correctly
        if tree1 is self.tree_start:
            full_path = path1[::-1] + path2
        else:
            full_path = path2[::-1] + path1

        # Compute path cost
        cost = self._compute_path_cost(full_path)

        if cost < self.best_cost:
            self.best_path = full_path
            self.best_cost = cost
            logger.debug(f"Improved joint path found! Cost: {cost:.4f}")

    def _rewire_tree(self, tree: Dict, new_idx: int):
        new_config = tree['configs'][new_idx]
        new_cost = tree['costs'][new_idx]

        # Find nearby configurations
        configs_array = np.array(tree['configs'])
        distances = np.linalg.norm(configs_array - new_config, axis=1)
        nearby_indices = np.where(distances < self.rewire_radius)[0]

        for idx in nearby_indices:
            if idx == new_idx:
                continue

            neighbor_config = tree['configs'][idx]
            potential_cost = new_cost + np.linalg.norm(neighbor_config - new_config)

            if (potential_cost < tree['costs'][idx] and 
                self._is_path_valid(new_config, neighbor_config)):
                tree['parents'][idx] = new_idx
                tree['costs'][idx] = potential_cost

    def _extract_path(self, tree: Dict, node_idx: int) -> JointPath:
        path = []
        current = node_idx
        while current != -1:
            path.append(tree['configs'][current])
            current = tree['parents'][current]
        return path

    def _compute_path_cost(self, path: JointPath) -> float:
        if len(path) < 2:
            return 0.0
        return sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))

    def _is_config_valid(self, config: JointConfiguration) -> bool:
        # Check joint limits with small margin
        margin = 0.01  # 0.01 rad margin
        if np.any(config < (self.joint_limits_lower + margin)) or np.any(config > (self.joint_limits_upper - margin)):
            return False

        # Basic workspace check
        try:
            T = self.robot_controller.forward_kinematics(config)
            position = T[:3, 3]
            if not self.robot._check_workspace(position):
                return False
        except Exception:
            return False

        return True

    def _is_path_valid(self, config1: JointConfiguration, config2: JointConfiguration, num_steps: int = 10) -> bool:
        if np.allclose(config1, config2):
            return True

        # Interpolate between configurations
        steps = max(1, min(6, num_steps))
        for i in range(steps + 1):
            alpha = i / steps
            interp_config = (1 - alpha) * config1 + alpha * config2
            if not self._is_config_valid(interp_config):
                return False
        return True


class MotionPlanner:
    """High-level motion planning interface with production-ready unit handling."""
    
    def __init__(self, robot_controller: RobotController):
        self.robot_controller = robot_controller
        self.unit_handler = ProductionUnitHandler(robot_controller)
        self.robot_env = RobotEnvironment(robot_controller)
        self.joint_space_planner = JointSpacePlanner(robot_controller)
        # Initialize Cartesian planner with production-grade parameters
        self.cartesian_path_planner = CartesianPathPlanner(
            self.robot_env, 
            robot_controller,
            max_iterations=2000,    # High-quality planning
            step_size=0.08,         # Precision steps for production
            goal_bias=0.12,         # Balanced exploration/exploitation
            connect_threshold=0.12, # Tight connections for smooth paths
            rewire_radius=0.2       # Optimal path optimization
        )
        
        logger.info("MotionPlanner initialized with production unit handling")
        
    def plan_joint_path(self, start_config: JointConfiguration, goal_config: JointConfiguration) -> Optional[JointPath]:
        """
        Plan a path in joint space.
        """
        return self.joint_space_planner.plan(start_config, goal_config)
        
    def plan_cartesian_path(self, start_pose: Union[CartesianPose, JointConfiguration], 
                            goal_pose: Union[CartesianPose, JointConfiguration]) -> Optional[Tuple[CartesianPath, JointPath]]:
        """
        Plan a path in Cartesian space.
        
        Args:
            start_pose: Start pose (4x4 matrix or [x,y,z,rx,ry,rz]) or start joint config
            goal_pose: Goal pose (4x4 matrix or [x,y,z,rx,ry,rz]) or goal joint config
            
        Returns:
            Tuple of (CartesianPath, JointPath) if successful, None otherwise
        """
        # Convert start_pose and goal_pose to 4x4 matrices if they are joint configs
        if isinstance(start_pose, np.ndarray) and start_pose.shape == (self.robot_controller.robot.n_joints,):
            start_pose_matrix = self.robot_controller.forward_kinematics(start_pose)
        else:
            start_pose_matrix = self._convert_to_matrix(start_pose)

        if isinstance(goal_pose, np.ndarray) and goal_pose.shape == (self.robot_controller.robot.n_joints,):
            goal_pose_matrix = self.robot_controller.forward_kinematics(goal_pose)
        else:
            goal_pose_matrix = self._convert_to_matrix(goal_pose)

        if start_pose_matrix is None or goal_pose_matrix is None:
            logger.error("Invalid start or goal pose provided for Cartesian planning.")
            return None

        start_pos = start_pose_matrix[:3, 3]
        goal_pos = goal_pose_matrix[:3, 3]

        # Plan geometric path in 3D Cartesian space
        cartesian_waypoints_pos = self.cartesian_path_planner.plan(start_pos, goal_pos)
        
        if cartesian_waypoints_pos is None:
            logger.warning("Failed to find Cartesian path.")
            return None
            
        # Interpolate orientation along the path
        cartesian_path_full_pose = self._interpolate_orientation(cartesian_waypoints_pos, start_pose_matrix, goal_pose_matrix)

        # Convert Cartesian path to joint space path
        joint_path = self._cartesian_to_joint_path(cartesian_path_full_pose)
        
        if joint_path is None:
            logger.warning("Failed to convert Cartesian path to joint space.")
            return None
            
        return cartesian_path_full_pose, joint_path
        
    def _convert_to_matrix(self, pose: Union[CartesianPose, np.ndarray]) -> Optional[np.ndarray]:
        """
        Helper to convert various pose formats to a 4x4 transformation matrix.
        Assumes [x, y, z, rx, ry, rz] for 6-element array where rx,ry,rz are RPY in radians.
        """
        if isinstance(pose, np.ndarray):
            if pose.shape == (4, 4):
                return pose
            elif pose.shape == (6,):
                # Assume [x, y, z, rx, ry, rz]
                T = np.eye(4)
                T[:3, 3] = pose[:3]
                T[:3, :3] = self.robot_controller.robot.rpy_to_matrix(pose[3:])
                return T
                
        logger.error(f"Unsupported pose format: {type(pose)} with shape {getattr(pose, 'shape', 'N/A')}")
        return None
        
    def _interpolate_orientation(self, waypoints: List[np.ndarray], start_pose: np.ndarray, goal_pose: np.ndarray) -> List[np.ndarray]:
        """Interpolate orientation along Cartesian path."""
        cartesian_path = []
        
        # Extract initial and final RPY angles
        start_rpy = self.robot_controller.robot.matrix_to_rpy(start_pose[:3, :3])
        goal_rpy = self.robot_controller.robot.matrix_to_rpy(goal_pose[:3, :3])

        for i, waypoint_pos in enumerate(waypoints):
            # Linear interpolation factor
            alpha = i / (len(waypoints) - 1) if len(waypoints) > 1 else 0
            
            # Interpolate orientation (simplified SLERP for RPY)
            # Handle angle wrapping for shortest path in RPY
            interp_rpy = start_rpy + alpha * (np.mod(goal_rpy - start_rpy + np.pi, 2*np.pi) - np.pi)
            interp_rotation = self.robot_controller.robot.rpy_to_matrix(interp_rpy)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, 3] = waypoint_pos
            T[:3, :3] = interp_rotation
            
            cartesian_path.append(T)
            
        return cartesian_path
        
    def _cartesian_to_joint_path(self, cartesian_path: List[np.ndarray]) -> Optional[JointPath]:
        """
        Convert Cartesian path to joint space path with proper unit handling.
        
        Args:
            cartesian_path: List of 4x4 transformation matrices in planning units (meters)
            
        Returns:
            Joint path in planning units (radians) or None if conversion fails
        """
        joint_path = []
        previous_config = None
        
        logger.info(f"Converting Cartesian path ({len(cartesian_path)} waypoints) to joint space")
        
        # Set enhanced IK parameters once at the beginning
        original_params = self.robot_controller.ik_params.copy()
        enhanced_params = {
            'max_iters': int(min(500, original_params.get('max_iters', 300) * 1.5)),
            'num_attempts': int(min(40, original_params.get('num_attempts', 30) * 1.3)),
        }
        self.robot_controller.set_ik_parameters(**enhanced_params)
        
        try:
            for i, pose_planning in enumerate(cartesian_path):
                # Use unit-aware IK solver
                q_solution, converged = self.unit_handler.inverse_kinematics_planning_units(
                    pose_planning, q_init_rad=previous_config)
                
                if (not converged) or (q_solution is None):
                    if i % 20 == 0:  # Only log every 20 failures to reduce noise
                        logger.debug(f"IK failed for waypoint {i}, trying fallback")
                    
                    # Generate small perturbations around pose (simplified approach)
                    pos = pose_planning[:3, 3].copy()
                    R = pose_planning[:3, :3].copy()
                    rpy = self.robot_controller.robot.matrix_to_rpy(R)
                    
                    # Try fewer, more targeted perturbations
                    perturbations = [
                        (np.array([0.003, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
                        (np.array([-0.003, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
                        (np.array([0.0, 0.003, 0.0]), np.array([0.0, 0.0, 0.0])),
                        (np.array([0.0, 0.0, 0.003]), np.array([0.0, 0.0, 0.0])),
                        (np.array([0.0, 0.0, 0.0]), np.array([0.03, 0.0, 0.0])),
                        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.03, 0.0])),
                    ]

                    success = False
                    for dp, drpy in perturbations:
                        T_try = np.eye(4)
                        T_try[:3, 3] = pos + dp
                        T_try[:3, :3] = self.robot_controller.robot.rpy_to_matrix(rpy + drpy)
                        q_try, ok = self.unit_handler.inverse_kinematics_planning_units(T_try, q_init_rad=previous_config)
                        if ok and q_try is not None:
                            q_solution, converged = q_try, True
                            success = True
                            pose_planning = T_try
                            break
                    
                    if not success:
                        # Last resort: subdivide segment (only if we have a previous pose)
                        if i > 0:
                            prev_pose = cartesian_path[i-1]
                            subdivided = self._solve_segment_with_subdivision(prev_pose, pose_planning, previous_config, max_depth=2)
                            if subdivided is None:
                                logger.error(f"IK failed for waypoint {i} after all fallbacks")
                                return None
                            # Append the successful intermediate configs
                            for q_sub in subdivided[:-1]:
                                joint_path.append(q_sub)
                            q_solution = subdivided[-1]
                            converged = True
                        else:
                            logger.error(f"IK failed for start waypoint {i}")
                            return None
                
                joint_path.append(q_solution)
                previous_config = q_solution
                
        finally:
            # Restore original IK parameters
            self.robot_controller.set_ik_parameters(**original_params)
            
        return joint_path

    def _solve_segment_with_subdivision(self, pose_a: np.ndarray, pose_b: np.ndarray,
                                        q_init: Optional[np.ndarray], max_depth: int = 2) -> Optional[List[np.ndarray]]:
        """Recursively subdivide the segment between two Cartesian poses to find IK-feasible steps.

        Returns list of joint configs from just after pose_a up to pose_b if successful.
        """
        # Try direct with unit-aware IK
        q_b, ok = self.unit_handler.inverse_kinematics_planning_units(pose_b, q_init_rad=q_init)
        if ok and q_b is not None:
            return [q_b]

        if max_depth <= 0:
            return None

        # Mid pose interpolation (position + orientation in RPY space)
        pos_a = pose_a[:3, 3]
        pos_b = pose_b[:3, 3]
        rpy_a = self.robot_controller.robot.matrix_to_rpy(pose_a[:3, :3])
        rpy_b = self.robot_controller.robot.matrix_to_rpy(pose_b[:3, :3])
        # shortest angle diff
        angle_diff = np.mod(rpy_b - rpy_a + np.pi, 2*np.pi) - np.pi
        rpy_mid = rpy_a + 0.5 * angle_diff
        pos_mid = 0.5 * (pos_a + pos_b)
        T_mid = np.eye(4)
        T_mid[:3, 3] = pos_mid
        T_mid[:3, :3] = self.robot_controller.robot.rpy_to_matrix(rpy_mid)

        # Solve for mid with unit-aware IK
        q_mid, ok_mid = self.unit_handler.inverse_kinematics_planning_units(T_mid, q_init_rad=q_init)
        if not ok_mid or q_mid is None:
            # Recurse on a->mid first
            left = self._solve_segment_with_subdivision(pose_a, T_mid, q_init, max_depth-1)
            if left is None:
                return None
            # Then mid->b using the last solution as seed
            right = self._solve_segment_with_subdivision(T_mid, pose_b, left[-1], max_depth-1)
            if right is None:
                return None
            return left + right
        else:
            # Solve mid->b using q_mid as seed
            tail = self._solve_segment_with_subdivision(T_mid, pose_b, q_mid, max_depth-1)
            if tail is None:
                # Try a->mid tail-first
                head = self._solve_segment_with_subdivision(pose_a, T_mid, q_init, max_depth-1)
                if head is None:
                    return None
                return head
            return [q_mid] + tail

    # ==========================================================================
    # ROBOT API INTERFACE METHODS (Degrees and Millimeters)
    # ==========================================================================
    
    def plan_joint_path_robot_units(self, start_joints_deg: np.ndarray, goal_joints_deg: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Plan joint space path using robot API units (degrees).
        
        Args:
            start_joints_deg: Start joint angles in degrees [6 joints]
            goal_joints_deg: Goal joint angles in degrees [6 joints]
            
        Returns:
            List of joint configurations in degrees, or None if planning fails
        """
        # Convert degrees to radians for internal planning
        start_joints_rad = np.deg2rad(start_joints_deg)
        goal_joints_rad = np.deg2rad(goal_joints_deg)
        
        # Plan in internal units (radians)
        joint_path_rad = self.plan_joint_path(start_joints_rad, goal_joints_rad)
        
        if joint_path_rad is None:
            return None
            
        # Convert back to degrees for robot API
        joint_path_deg = [np.rad2deg(q) for q in joint_path_rad]
        
        logger.info(f"Joint path planned: {len(joint_path_deg)} waypoints in degrees")
        return joint_path_deg
    
    def plan_cartesian_path_robot_units(self, start_pose_robot: np.ndarray, goal_pose_robot: np.ndarray) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Plan Cartesian space path using robot API units (mm and degrees).
        
        Args:
            start_pose_robot: Start pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            goal_pose_robot: Goal pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            
        Returns:
            Tuple of (cartesian_path_robot_units, joint_path_robot_units) or None if planning fails
            - cartesian_path_robot_units: List of poses in robot units [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            - joint_path_robot_units: List of joint configurations in degrees
        """
        # Convert robot units to internal planning units
        start_pose_planning = self._robot_pose_to_planning_units(start_pose_robot)
        goal_pose_planning = self._robot_pose_to_planning_units(goal_pose_robot)
        
        # Plan in internal units
        result = self.plan_cartesian_path(start_pose_planning, goal_pose_planning)
        
        if result is None:
            return None
            
        cartesian_path_planning, joint_path_rad = result
        
        # Convert back to robot units
        cartesian_path_robot = [self._planning_pose_to_robot_units(T) for T in cartesian_path_planning]
        joint_path_deg = [np.rad2deg(q) for q in joint_path_rad]
        
        logger.info(f"Cartesian path planned: {len(cartesian_path_robot)} cartesian waypoints, {len(joint_path_deg)} joint waypoints in robot units")
        return cartesian_path_robot, joint_path_deg
    
    def move_joint_robot_api(self, target_joints_deg: np.ndarray, current_joints_deg: np.ndarray = None) -> Optional[List[np.ndarray]]:
        """
        Plan path to target joint configuration using robot API units (compatible with move_joint() API).
        
        Args:
            target_joints_deg: Target joint angles in degrees [6 joints]
            current_joints_deg: Current joint angles in degrees [6 joints]. If None, uses robot's current position
            
        Returns:
            List of joint configurations in degrees for robot execution, or None if planning fails
        """
        if current_joints_deg is None:
            # In a real implementation, this would get current joints from robot
            # For now, assume zero position
            current_joints_deg = np.zeros(6)
            logger.warning("Current joint position not provided, assuming zero position")
        
        return self.plan_joint_path_robot_units(current_joints_deg, target_joints_deg)
    
    def move_blend_point_robot_api(self, target_pose_robot: np.ndarray, current_pose_robot: np.ndarray = None) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Plan path to target Cartesian pose using robot API units (compatible with move_blend_point() API).
        
        Args:
            target_pose_robot: Target pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            current_pose_robot: Current pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]. If None, uses robot's current position
            
        Returns:
            Tuple of (cartesian_path_robot_units, joint_path_robot_units) or None if planning fails
        """
        if current_pose_robot is None:
            # In a real implementation, this would get current pose from robot
            # For now, assume a default pose
            current_pose_robot = np.array([0.0, -6.5, 877.0, 0.0, 0.0, 0.0])  # Example from JSON data
            logger.warning("Current pose not provided, assuming default pose")
        
        return self.plan_cartesian_path_robot_units(current_pose_robot, target_pose_robot)
    
    def validate_robot_data_format(self, json_data_path: str) -> Dict[str, bool]:
        """
        Validate that the planning system can handle real robot data format.
        
        Args:
            json_data_path: Path to JSON file with real robot data
            
        Returns:
            Dictionary with validation results
        """
        try:
            import json
            with open(json_data_path, 'r') as f:
                data = json.load(f)
            
            results = {
                'json_loaded': True,
                'has_waypoints': 'waypoints' in data,
                'has_joint_positions': False,
                'has_tcp_position': False,
                'joint_units_valid': False,
                'tcp_units_valid': False
            }
            
            if results['has_waypoints'] and len(data['waypoints']) > 0:
                first_waypoint = data['waypoints'][0]
                
                # Check joint positions
                if 'joint_positions' in first_waypoint:
                    results['has_joint_positions'] = True
                    joints = np.array(first_waypoint['joint_positions'])
                    # Real robot joints should be in reasonable degree range
                    results['joint_units_valid'] = np.all(np.abs(joints) < 360)  # Reasonable degree range
                
                # Check TCP position  
                if 'tcp_position' in first_waypoint:
                    results['has_tcp_position'] = True
                    tcp = np.array(first_waypoint['tcp_position'])
                    # Real robot TCP should be in mm range (positions > 100, < 2000)
                    results['tcp_units_valid'] = (len(tcp) >= 6 and 
                                                tcp[2] > 100 and tcp[2] < 2000)  # Z position in mm range
            
            logger.info(f"Robot data validation: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to validate robot data: {e}")
            return {'error': str(e)}
    
    # ==========================================================================
    # INTERNAL UNIT CONVERSION HELPERS
    # ==========================================================================
    
    def _robot_pose_to_planning_units(self, pose_robot: np.ndarray) -> np.ndarray:
        """Convert robot API pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] to planning units [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]."""
        pose_planning = pose_robot.copy()
        pose_planning[:3] = pose_robot[:3] / 1000.0  # mm to meters
        pose_planning[3:] = np.deg2rad(pose_robot[3:])  # degrees to radians
        return pose_planning
    
    def _planning_pose_to_robot_units(self, T_planning: np.ndarray) -> np.ndarray:
        """Convert planning 4x4 matrix to robot API format [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        # Extract position (convert m to mm)
        pos_mm = T_planning[:3, 3] * 1000.0
        
        # Extract rotation (convert to RPY in degrees)
        R = T_planning[:3, :3]
        rpy_rad = self.robot_controller.robot.matrix_to_rpy(R)
        rpy_deg = np.rad2deg(rpy_rad)
        
        return np.concatenate([pos_mm, rpy_deg])


