"""
Unit conversion utilities for robot planning system.

This module handles conversions between:
- Planning system: SI units (meters, radians)
- Robot kinematics: Robot units (mm, degrees)

Production-ready unit handling for robot manipulator planning.
"""

import numpy as np
import logging
from typing import Union, Tuple, List
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

class UnitConverter:
    """Handles unit conversions between planning system and robot kinematics."""
    
    # Unit conversion constants
    M_TO_MM = 1000.0
    MM_TO_M = 1.0 / 1000.0
    RAD_TO_DEG = 180.0 / np.pi
    DEG_TO_RAD = np.pi / 180.0
    
    @staticmethod
    def planning_to_robot_position(pos_m: np.ndarray) -> np.ndarray:
        """
        Convert position from planning units (meters) to robot units (mm).
        
        Args:
            pos_m: Position in meters [x, y, z]
            
        Returns:
            Position in millimeters [x_mm, y_mm, z_mm]
        """
        return pos_m * UnitConverter.M_TO_MM
    
    @staticmethod
    def robot_to_planning_position(pos_mm: np.ndarray) -> np.ndarray:
        """
        Convert position from robot units (mm) to planning units (meters).
        
        Args:
            pos_mm: Position in millimeters [x_mm, y_mm, z_mm]
            
        Returns:
            Position in meters [x, y, z]
        """
        return pos_mm * UnitConverter.MM_TO_M
    
    @staticmethod
    def planning_to_robot_angles(angles_rad: np.ndarray) -> np.ndarray:
        """
        Convert angles from planning units (radians) to robot units (degrees).
        
        Args:
            angles_rad: Angles in radians
            
        Returns:
            Angles in degrees
        """
        return angles_rad * UnitConverter.RAD_TO_DEG
    
    @staticmethod
    def robot_to_planning_angles(angles_deg: np.ndarray) -> np.ndarray:
        """
        Convert angles from robot units (degrees) to planning units (radians).
        
        Args:
            angles_deg: Angles in degrees
            
        Returns:
            Angles in radians
        """
        return angles_deg * UnitConverter.DEG_TO_RAD
    
    @staticmethod
    def planning_to_robot_transform(T_planning: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert homogeneous transformation from planning units to robot units.
        
        Args:
            T_planning: 4x4 transformation matrix in planning units (meters, radians)
            
        Returns:
            Tuple of (position_mm, rpy_deg) for robot kinematics
        """
        # Extract position and convert to mm
        pos_m = T_planning[:3, 3]
        pos_mm = UnitConverter.planning_to_robot_position(pos_m)
        
        # Extract rotation and convert to RPY degrees
        R = T_planning[:3, :3]
        rpy_rad = Rotation.from_matrix(R).as_euler('xyz', degrees=False)
        rpy_deg = UnitConverter.planning_to_robot_angles(rpy_rad)
        
        return pos_mm, rpy_deg
    
    @staticmethod
    def robot_to_planning_transform(pos_mm: np.ndarray, rpy_deg: np.ndarray) -> np.ndarray:
        """
        Convert robot units to homogeneous transformation in planning units.
        
        Args:
            pos_mm: Position in millimeters [x_mm, y_mm, z_mm]
            rpy_deg: RPY angles in degrees [roll, pitch, yaw]
            
        Returns:
            4x4 transformation matrix in planning units (meters, radians)
        """
        # Convert position to meters
        pos_m = UnitConverter.robot_to_planning_position(pos_mm)
        
        # Convert RPY to rotation matrix
        rpy_rad = UnitConverter.robot_to_planning_angles(rpy_deg)
        R = Rotation.from_euler('xyz', rpy_rad, degrees=False).as_matrix()
        
        # Construct transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos_m
        
        return T
    
    @staticmethod
    def planning_pose_to_robot_format(T_planning: np.ndarray) -> np.ndarray:
        """
        Convert planning pose to robot controller format.
        
        Args:
            T_planning: 4x4 transformation matrix in planning units
            
        Returns:
            6-element array [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
        """
        pos_mm, rpy_deg = UnitConverter.planning_to_robot_transform(T_planning)
        return np.concatenate([pos_mm, rpy_deg])
    
    @staticmethod
    def robot_format_to_planning_pose(pose_robot: np.ndarray) -> np.ndarray:
        """
        Convert robot controller format to planning pose.
        
        Args:
            pose_robot: 6-element array [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
            
        Returns:
            4x4 transformation matrix in planning units
        """
        pos_mm = pose_robot[:3]
        rpy_deg = pose_robot[3:]
        return UnitConverter.robot_to_planning_transform(pos_mm, rpy_deg)
    
    @staticmethod
    def validate_planning_workspace(pos_m: np.ndarray, 
                                  workspace_limits_mm: Tuple[Tuple[float, float], ...]) -> bool:
        """
        Validate if planning position is within robot workspace limits.
        
        Args:
            pos_m: Position in planning units (meters)
            workspace_limits_mm: Workspace limits in robot units (mm)
                                 ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            
        Returns:
            True if position is within workspace
        """
        pos_mm = UnitConverter.planning_to_robot_position(pos_m)
        
        for i, (min_val, max_val) in enumerate(workspace_limits_mm):
            if not (min_val <= pos_mm[i] <= max_val):
                logger.warning(f"Position {pos_mm} outside workspace limits {workspace_limits_mm}")
                return False
        
        return True
    
    @staticmethod
    def convert_path_planning_to_robot(cartesian_path_planning: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert entire Cartesian path from planning to robot units.
        
        Args:
            cartesian_path_planning: List of 4x4 transformation matrices in planning units
            
        Returns:
            List of 6-element pose arrays in robot units
        """
        robot_path = []
        for T_planning in cartesian_path_planning:
            pose_robot = UnitConverter.planning_pose_to_robot_format(T_planning)
            robot_path.append(pose_robot)
        
        return robot_path
    
    @staticmethod
    def convert_path_robot_to_planning(robot_path: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert entire path from robot to planning units.
        
        Args:
            robot_path: List of 6-element pose arrays in robot units
            
        Returns:
            List of 4x4 transformation matrices in planning units
        """
        planning_path = []
        for pose_robot in robot_path:
            T_planning = UnitConverter.robot_format_to_planning_pose(pose_robot)
            planning_path.append(T_planning)
        
        return planning_path
    
    @staticmethod
    def log_conversion_info(operation: str, input_units: str, output_units: str, 
                          input_value: Union[float, np.ndarray], 
                          output_value: Union[float, np.ndarray]):
        """
        Log unit conversion information for debugging.
        
        Args:
            operation: Description of the conversion operation
            input_units: Input unit description
            output_units: Output unit description
            input_value: Input value
            output_value: Output value
        """
        logger.debug(f"Unit conversion - {operation}")
        logger.debug(f"  Input ({input_units}): {input_value}")
        logger.debug(f"  Output ({output_units}): {output_value}")


class ProductionUnitHandler:
    """Production-ready unit handling for robot planning system."""
    
    def __init__(self, robot_controller):
        """
        Initialize with robot controller for unit-aware operations.
        
        Args:
            robot_controller: Robot controller instance with unit conversion methods
        """
        self.robot_controller = robot_controller
        self.converter = UnitConverter()
        
        # Robot workspace limits in mm (example - should be configured per robot)
        self.workspace_limits_mm = (
            (-800, 800),   # X limits in mm
            (-800, 800),   # Y limits in mm
            (0, 1200)      # Z limits in mm
        )
        
        logger.info("Production unit handler initialized")
    
    def forward_kinematics_planning_units(self, q_rad: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics with planning units input/output.
        
        Args:
            q_rad: Joint angles in radians (planning units)
            
        Returns:
            4x4 transformation matrix in planning units (meters)
        """
        # Robot controller expects radians, returns meters - direct call
        return self.robot_controller.forward_kinematics(q_rad)
    
    def inverse_kinematics_planning_units(self, T_planning: np.ndarray, 
                                        q_init_rad: np.ndarray = None) -> Tuple[np.ndarray, bool]:
        """
        Compute inverse kinematics with planning units input/output.
        
        Args:
            T_planning: Target pose in planning units (4x4 matrix, meters)
            q_init_rad: Initial joint configuration in radians
            
        Returns:
            Tuple of (joint_angles_rad, success)
        """
        # Validate workspace
        pos_m = T_planning[:3, 3]
        if not self.converter.validate_planning_workspace(pos_m, self.workspace_limits_mm):
            logger.warning("Target pose outside robot workspace")
            return None, False
        
        # Robot controller expects and returns radians - direct call
        return self.robot_controller.inverse_kinematics(T_planning, q_init_rad)
    
    def validate_joint_limits_planning_units(self, q_rad: np.ndarray) -> bool:
        """
        Validate joint limits using planning units.
        
        Args:
            q_rad: Joint angles in radians
            
        Returns:
            True if within limits
        """
        # Get joint limits in radians from robot
        joint_limits_rad = self.robot_controller.robot.joint_limits
        
        for i, (q, (q_min, q_max)) in enumerate(zip(q_rad, joint_limits_rad)):
            if not (q_min <= q <= q_max):
                logger.warning(f"Joint {i} angle {np.rad2deg(q):.1f}° outside limits "
                             f"[{np.rad2deg(q_min):.1f}°, {np.rad2deg(q_max):.1f}°]")
                return False
        
        return True
    
    def get_workspace_bounds_planning_units(self) -> Tuple[Tuple[float, float], ...]:
        """
        Get workspace bounds in planning units (meters).
        
        Returns:
            Workspace bounds ((x_min_m, x_max_m), (y_min_m, y_max_m), (z_min_m, z_max_m))
        """
        bounds_m = []
        for min_mm, max_mm in self.workspace_limits_mm:
            min_m = min_mm * self.converter.MM_TO_M
            max_m = max_mm * self.converter.MM_TO_M
            bounds_m.append((min_m, max_m))
        
        return tuple(bounds_m)

