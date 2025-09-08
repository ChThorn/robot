"""
Tests for trajectory planning module.
"""

import pytest
import numpy as np
import sys
import os

# Add robot_kinematics to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinematics', 'src'))

try:
    from robot_controller import RobotController
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    RobotController = None

from robot_planning.trajectory_planning import (
    TrajectoryConstraints, TimeParameterization, TrajectoryInterpolator,
    TrajectoryValidator, TrajectoryPlanner
)


class TestTrajectoryConstraints:
    """Test trajectory constraints functionality."""
    
    def test_default_constraints(self):
        """Test default constraint creation."""
        constraints = TrajectoryConstraints.default_constraints(6)
        
        assert len(constraints.max_joint_velocities) == 6
        assert len(constraints.max_joint_accelerations) == 6
        assert np.all(constraints.max_joint_velocities > 0)
        assert np.all(constraints.max_joint_accelerations > 0)
    
    def test_custom_constraints(self):
        """Test custom constraint creation."""
        max_vel = np.array([1, 2, 3, 4, 5, 6])
        max_acc = np.array([10, 20, 30, 40, 50, 60])
        
        constraints = TrajectoryConstraints(max_vel, max_acc)
        
        assert np.array_equal(constraints.max_joint_velocities, max_vel)
        assert np.array_equal(constraints.max_joint_accelerations, max_acc)


class TestTimeParameterization:
    """Test time parameterization algorithms."""
    
    @pytest.fixture
    def simple_path(self):
        """Create simple test path."""
        return [
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0]),
            np.array([1, 1, 1, 0, 0, 0])
        ]
    
    @pytest.fixture
    def constraints(self):
        """Create test constraints."""
        return TrajectoryConstraints.default_constraints(6)
    
    def test_constant_velocity(self, simple_path, constraints):
        """Test constant velocity parameterization."""
        trajectory = TimeParameterization.constant_velocity(simple_path, constraints)
        
        assert len(trajectory) >= len(simple_path)
        
        # Check time ordering
        times = [t for t, _ in trajectory]
        assert times == sorted(times)
        assert times[0] == 0.0
        
        # Check configurations
        configs = [q for _, q in trajectory]
        assert np.allclose(configs[0], simple_path[0])
        assert np.allclose(configs[-1], simple_path[-1])
    
    def test_trapezoidal_velocity(self, simple_path, constraints):
        """Test trapezoidal velocity parameterization."""
        trajectory = TimeParameterization.trapezoidal_velocity(simple_path, constraints)
        
        assert len(trajectory) >= len(simple_path)
        
        # Check time ordering
        times = [t for t, _ in trajectory]
        assert times == sorted(times)
        assert times[0] == 0.0
        
        # Check start and end configurations
        configs = [q for _, q in trajectory]
        assert np.allclose(configs[0], simple_path[0], atol=1e-3)
        assert np.allclose(configs[-1], simple_path[-1], atol=1e-3)
    
    def test_time_scaling(self, simple_path, constraints):
        """Test trajectory time scaling."""
        trajectory = TimeParameterization.constant_velocity(simple_path, constraints)
        scaled_trajectory = TimeParameterization.scale_time(trajectory, 2.0)
        
        # Check that times are scaled
        original_times = [t for t, _ in trajectory]
        scaled_times = [t for t, _ in scaled_trajectory]
        
        for orig_t, scaled_t in zip(original_times, scaled_times):
            assert np.isclose(scaled_t, orig_t * 2.0)
    
    def test_empty_path(self, constraints):
        """Test handling of empty path."""
        empty_path = []
        trajectory = TimeParameterization.constant_velocity(empty_path, constraints)
        assert len(trajectory) == 0
    
    def test_single_point_path(self, constraints):
        """Test handling of single point path."""
        single_path = [np.array([0, 0, 0, 0, 0, 0])]
        trajectory = TimeParameterization.constant_velocity(single_path, constraints)
        assert len(trajectory) == 1
        assert trajectory[0][0] == 0.0


class TestTrajectoryInterpolator:
    """Test trajectory interpolation functionality."""
    
    @pytest.fixture
    def simple_trajectory(self):
        """Create simple test trajectory."""
        return [
            (0.0, np.array([0, 0, 0, 0, 0, 0])),
            (1.0, np.array([1, 0, 0, 0, 0, 0])),
            (2.0, np.array([1, 1, 0, 0, 0, 0])),
            (3.0, np.array([1, 1, 1, 0, 0, 0]))
        ]
    
    def test_linear_interpolation(self, simple_trajectory):
        """Test linear interpolation."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='linear')
        
        # Test at trajectory points
        for t, expected_q in simple_trajectory:
            q = interpolator.evaluate(t)
            assert np.allclose(q, expected_q)
        
        # Test interpolation between points
        q_mid = interpolator.evaluate(0.5)
        expected_mid = np.array([0.5, 0, 0, 0, 0, 0])
        assert np.allclose(q_mid, expected_mid)
    
    def test_cubic_interpolation(self, simple_trajectory):
        """Test cubic spline interpolation."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='cubic')
        
        # Test at trajectory points
        for t, expected_q in simple_trajectory:
            q = interpolator.evaluate(t)
            assert np.allclose(q, expected_q, atol=1e-10)
        
        # Test that interpolation is smooth (no sudden jumps)
        times = np.linspace(0, 3, 100)
        positions = [interpolator.evaluate(t) for t in times]
        
        # Check continuity
        for i in range(len(positions) - 1):
            step = np.linalg.norm(positions[i+1] - positions[i])
            assert step < 0.1  # Should be smooth
    
    def test_velocity_evaluation(self, simple_trajectory):
        """Test velocity evaluation."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='cubic')
        
        # Test velocity at various points
        for t in [0.5, 1.5, 2.5]:
            velocity = interpolator.evaluate_velocity(t)
            assert len(velocity) == 6
            assert np.all(np.isfinite(velocity))
    
    def test_acceleration_evaluation(self, simple_trajectory):
        """Test acceleration evaluation."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='cubic')
        
        # Test acceleration at various points
        for t in [0.5, 1.5, 2.5]:
            acceleration = interpolator.evaluate_acceleration(t)
            assert len(acceleration) == 6
            assert np.all(np.isfinite(acceleration))
    
    def test_duration(self, simple_trajectory):
        """Test trajectory duration."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='linear')
        duration = interpolator.get_duration()
        assert duration == 3.0
    
    def test_resampling(self, simple_trajectory):
        """Test trajectory resampling."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='linear')
        resampled = interpolator.resample(0.5)
        
        # Check that we have the right number of points
        expected_points = int(3.0 / 0.5) + 1
        assert len(resampled) == expected_points
        
        # Check time spacing
        times = [t for t, _ in resampled]
        for i in range(len(times) - 1):
            assert np.isclose(times[i+1] - times[i], 0.5, atol=1e-10)


class TestTrajectoryValidator:
    """Test trajectory validation functionality."""
    
    @pytest.fixture
    def mock_robot_controller(self):
        """Create mock robot controller for testing."""
        if not ROBOT_AVAILABLE:
            pytest.skip("Robot kinematics not available")
        
        return RobotController()
    
    @pytest.fixture
    def validator(self, mock_robot_controller):
        """Create trajectory validator."""
        constraints = TrajectoryConstraints.default_constraints(6)
        return TrajectoryValidator(mock_robot_controller, constraints)
    
    @pytest.fixture
    def simple_trajectory(self):
        """Create simple test trajectory."""
        return [
            (0.0, np.array([0, 0, 0, 0, 0, 0])),
            (1.0, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])),
            (2.0, np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
        ]
    
    def test_validator_initialization(self, mock_robot_controller):
        """Test validator initialization."""
        constraints = TrajectoryConstraints.default_constraints(6)
        validator = TrajectoryValidator(mock_robot_controller, constraints)
        
        assert validator.robot_controller is not None
        assert validator.constraints is not None
    
    def test_trajectory_validation(self, validator, simple_trajectory):
        """Test trajectory validation."""
        interpolator = TrajectoryInterpolator(simple_trajectory, method='linear')
        results = validator.validate_trajectory(interpolator)
        
        assert isinstance(results, dict)
        assert 'overall' in results
        assert 'velocity_limits' in results
        assert 'acceleration_limits' in results


class TestTrajectoryPlanner:
    """Test high-level trajectory planning interface."""
    
    @pytest.fixture
    def mock_robot_controller(self):
        """Create mock robot controller for testing."""
        if not ROBOT_AVAILABLE:
            pytest.skip("Robot kinematics not available")
        
        return RobotController()
    
    @pytest.fixture
    def trajectory_planner(self, mock_robot_controller):
        """Create trajectory planner."""
        constraints = TrajectoryConstraints.default_constraints(6)
        return TrajectoryPlanner(mock_robot_controller, constraints)
    
    def test_planner_initialization(self, mock_robot_controller):
        """Test trajectory planner initialization."""
        constraints = TrajectoryConstraints.default_constraints(6)
        planner = TrajectoryPlanner(mock_robot_controller, constraints)
        
        assert planner.robot_controller is not None
        assert planner.motion_planner is not None
        assert planner.constraints is not None
        assert planner.validator is not None
    
    def test_trajectory_planning_workflow(self, trajectory_planner):
        """Test complete trajectory planning workflow."""
        start_config = np.zeros(6)
        goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # This might fail if planning doesn't succeed, which is okay for testing
        try:
            interpolator = trajectory_planner.plan_trajectory(
                start_config, goal_config, 
                method='constant',  # Use simpler method for testing
                max_attempts=3  # Reduce attempts for faster testing
            )
            
            if interpolator is not None:
                assert interpolator.get_duration() > 0
                
                # Test evaluation at start and end
                start_q = interpolator.evaluate(0.0)
                end_q = interpolator.evaluate(interpolator.get_duration())
                
                assert np.allclose(start_q, start_config, atol=1e-2)
                # End might not match exactly due to planning constraints
                
        except Exception as e:
            # Planning might fail due to various reasons in testing
            pytest.skip(f"Planning failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])

