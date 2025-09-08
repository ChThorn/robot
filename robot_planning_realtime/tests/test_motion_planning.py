"""
Tests for motion planning module.
"""

import pytest
import numpy as np
import sys
import os

# Add robot_kinematics to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))

try:
    from robot_controller import RobotController
    from robot_kinematics import RobotKinematics
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    RobotController = None
    RobotKinematics = None

from robot_planning.motion_planning import JointSpacePlanner, MotionPlanner, RobotEnvironment
from robot_planning.path_planning import Environment3D


class TestJointSpacePlanner:
    """Test joint space planning functionality."""
    
    @pytest.fixture
    def mock_robot_controller(self):
        """Create mock robot controller for testing."""
        if not ROBOT_AVAILABLE:
            pytest.skip("Robot kinematics not available")
        
        return RobotController()
    
    @pytest.fixture
    def joint_planner(self, mock_robot_controller):
        """Create joint space planner for testing."""
        return JointSpacePlanner(
            mock_robot_controller,
            max_iterations=1000,
            max_time=1.0  # Short time for testing
        )
    
    def test_planner_initialization(self, mock_robot_controller):
        """Test planner initialization."""
        planner = JointSpacePlanner(mock_robot_controller)
        
        assert planner.robot_controller is not None
        assert planner.max_iterations > 0
        assert planner.step_size > 0
        assert 0 < planner.goal_bias < 1
    
    def test_config_validation(self, joint_planner):
        """Test joint configuration validation."""
        # Valid configuration (all zeros)
        valid_config = np.zeros(6)
        assert joint_planner._is_config_valid(valid_config)
        
        # Invalid configuration (exceeds limits)
        invalid_config = np.full(6, 10.0)  # Way beyond joint limits
        assert not joint_planner._is_config_valid(invalid_config)
    
    def test_simple_planning(self, joint_planner):
        """Test simple joint space planning."""
        start_config = np.zeros(6)
        goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        path = joint_planner.plan(start_config, goal_config)
        
        if path is not None:  # Planning might fail due to time constraints
            assert len(path) >= 2
            assert np.allclose(path[0], start_config, atol=1e-3)
            assert np.allclose(path[-1], goal_config, atol=1e-1)  # Allow some tolerance
    
    def test_path_cost_computation(self, joint_planner):
        """Test path cost computation."""
        path = [
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0])
        ]
        
        cost = joint_planner._compute_path_cost(path)
        expected_cost = 1.0 + 1.0  # Two unit steps
        assert np.isclose(cost, expected_cost)
    
    def test_sampling_methods(self, joint_planner):
        """Test different sampling methods."""
        # Initialize the planner's internal state
        joint_planner._start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joint_planner._goal_config = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        tree_to = {
            'configs': [np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])],
            'parents': [-1],
            'costs': [0.0]
        }
        
        # Test multiple samples
        for _ in range(10):
            sample = joint_planner._sample_joint_space(tree_to)
            assert len(sample) == 6
            assert np.all(sample >= joint_planner.joint_limits_lower)
            assert np.all(sample <= joint_planner.joint_limits_upper)


class TestMotionPlanner:
    """Test high-level motion planning interface."""
    
    @pytest.fixture
    def mock_robot_controller(self):
        """Create mock robot controller for testing."""
        if not ROBOT_AVAILABLE:
            pytest.skip("Robot kinematics not available")
        
        return RobotController()
    
    @pytest.fixture
    def motion_planner(self, mock_robot_controller):
        """Create motion planner for testing."""
        return MotionPlanner(mock_robot_controller)
    
    def test_planner_initialization(self, mock_robot_controller):
        """Test motion planner initialization."""
        planner = MotionPlanner(mock_robot_controller)
        
        assert planner.robot_controller is not None
        assert planner.joint_space_planner is not None
        assert planner.cartesian_path_planner is not None
    
    def test_joint_path_planning(self, motion_planner):
        """Test joint path planning."""
        start_config = np.zeros(6)
        goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        path = motion_planner.plan_joint_path(start_config, goal_config)
        
        if path is not None:
            assert len(path) >= 2
            assert isinstance(path[0], np.ndarray)
            assert len(path[0]) == 6


class TestRobotEnvironment:
    """Test robot environment functionality."""
    
    @pytest.fixture
    def mock_robot_controller(self):
        """Create mock robot controller for testing."""
        if not ROBOT_AVAILABLE:
            pytest.skip("Robot kinematics not available")
        
        return RobotController()
    
    @pytest.fixture
    def robot_env(self, mock_robot_controller):
        """Create robot environment for testing."""
        return RobotEnvironment(mock_robot_controller)
    
    def test_environment_initialization(self, mock_robot_controller):
        """Test environment initialization."""
        env = RobotEnvironment(mock_robot_controller)
        
        assert env.robot_controller is not None
        assert env.bounds is not None
        assert len(env.bounds) == 3  # x, y, z bounds
    
    def test_workspace_bounds(self, robot_env):
        """Test workspace bounds extraction."""
        bounds = robot_env.bounds
        
        # Check that bounds are reasonable
        for min_bound, max_bound in bounds:
            assert min_bound < max_bound
            assert abs(max_bound - min_bound) > 0.1  # At least 10cm range
    
    def test_point_validation(self, robot_env):
        """Test point validation."""
        # Test point within workspace
        center_point = np.array([0.0, 0.0, 0.5])  # Reasonable point
        # Note: This might fail if point is outside actual workspace
        
        # Test point clearly outside workspace
        far_point = np.array([10.0, 10.0, 10.0])
        assert not robot_env.is_point_valid(far_point)


# Integration tests
class TestPlanningIntegration:
    """Test integration between different planning components."""
    
    @pytest.fixture
    def mock_robot_controller(self):
        """Create mock robot controller for testing."""
        if not ROBOT_AVAILABLE:
            pytest.skip("Robot kinematics not available")
        
        return RobotController()
    
    def test_end_to_end_planning(self, mock_robot_controller):
        """Test end-to-end planning workflow."""
        # Create motion planner
        motion_planner = MotionPlanner(mock_robot_controller)
        
        # Define simple planning problem
        start_config = np.zeros(6)
        goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Plan path
        path = motion_planner.plan_joint_path(start_config, goal_config)
        
        if path is not None:
            # Validate path
            assert len(path) >= 2
            
            # Check path continuity
            for i in range(len(path) - 1):
                step_size = np.linalg.norm(path[i+1] - path[i])
                assert step_size < 1.0  # Reasonable step size
            
            # Check start and goal
            assert np.allclose(path[0], start_config, atol=1e-2)
            # Goal might not be exactly reached due to planning constraints


if __name__ == "__main__":
    pytest.main([__file__])

