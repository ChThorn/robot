"""
Tests for utility functions.
"""

import pytest
import numpy as np
import tempfile
import os

from robot_planning.utils import (
    PlanningConfig, load_config, save_config,
    validate_joint_configuration, interpolate_joint_path,
    compute_path_length, smooth_joint_path
)


class TestPlanningConfig:
    """Test planning configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PlanningConfig()
        
        assert config.max_iterations > 0
        assert config.step_size > 0
        assert 0 < config.goal_bias < 1
        assert len(config.max_joint_velocities) == 6
        assert len(config.max_joint_accelerations) == 6
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'max_iterations': 1000,
            'step_size': 0.1,
            'goal_bias': 0.2,
            'max_joint_velocities': [1, 2, 3, 4, 5, 6],
            'max_joint_accelerations': [10, 20, 30, 40, 50, 60]
        }
        
        config = PlanningConfig.from_dict(config_dict)
        
        assert config.max_iterations == 1000
        assert config.step_size == 0.1
        assert config.goal_bias == 0.2
        assert np.array_equal(config.max_joint_velocities, [1, 2, 3, 4, 5, 6])
        assert np.array_equal(config.max_joint_accelerations, [10, 20, 30, 40, 50, 60])
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = PlanningConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'max_iterations' in config_dict
        assert 'max_joint_velocities' in config_dict
        assert isinstance(config_dict['max_joint_velocities'], list)


class TestConfigIO:
    """Test configuration file I/O."""
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = PlanningConfig()
        config.max_iterations = 1234
        config.step_size = 0.123
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            save_config(config, config_path)
            assert os.path.exists(config_path)
            
            # Load config
            loaded_config = load_config(config_path)
            
            assert loaded_config.max_iterations == 1234
            assert loaded_config.step_size == 0.123
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        config = load_config('/nonexistent/path/config.yaml')
        
        # Should return default config
        assert isinstance(config, PlanningConfig)
        assert config.max_iterations > 0


class TestJointValidation:
    """Test joint configuration validation."""
    
    @pytest.fixture
    def joint_limits(self):
        """Create test joint limits."""
        return np.array([
            [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi],  # Lower limits
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]          # Upper limits
        ])
    
    def test_valid_configuration(self, joint_limits):
        """Test validation of valid configuration."""
        valid_config = np.array([0, 0, 0, 0, 0, 0])
        assert validate_joint_configuration(valid_config, joint_limits)
        
        # Test configuration within limits
        valid_config2 = np.array([1, -1, 0.5, -0.5, 2, -2])
        assert validate_joint_configuration(valid_config2, joint_limits)
    
    def test_invalid_configuration(self, joint_limits):
        """Test validation of invalid configuration."""
        # Configuration exceeding upper limits
        invalid_config1 = np.array([4, 4, 4, 4, 4, 4])
        assert not validate_joint_configuration(invalid_config1, joint_limits)
        
        # Configuration below lower limits
        invalid_config2 = np.array([-4, -4, -4, -4, -4, -4])
        assert not validate_joint_configuration(invalid_config2, joint_limits)
    
    def test_margin_validation(self, joint_limits):
        """Test validation with safety margin."""
        # Configuration at the limit
        limit_config = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        
        # Should be invalid with default margin
        assert not validate_joint_configuration(limit_config, joint_limits)
        
        # Should be valid with zero margin
        assert validate_joint_configuration(limit_config, joint_limits, margin=0.0)
    
    def test_invalid_input(self, joint_limits):
        """Test validation with invalid input."""
        # Wrong shape - should handle gracefully
        invalid_input = np.array([1, 2, 3])  # Only 3 joints
        assert not validate_joint_configuration(invalid_input, joint_limits)
        
        # Not numpy array - should handle gracefully  
        invalid_input2 = [1, 2, 3, 4, 5, 6]
        assert not validate_joint_configuration(invalid_input2, joint_limits)


class TestPathUtilities:
    """Test path utility functions."""
    
    def test_interpolate_joint_path(self):
        """Test joint path interpolation."""
        start = np.array([0, 0, 0, 0, 0, 0])
        goal = np.array([1, 1, 1, 1, 1, 1])
        
        path = interpolate_joint_path(start, goal, num_points=5)
        
        assert path.shape == (5, 6)
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)
        
        # Check interpolation
        expected_mid = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert np.allclose(path[2], expected_mid)
    
    def test_compute_path_length(self):
        """Test path length computation."""
        # Simple path with known length
        path = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0]
        ])
        
        length = compute_path_length(path)
        expected_length = 1.0 + 1.0 + 1.0  # Three unit steps
        assert np.isclose(length, expected_length)
    
    def test_compute_path_length_empty(self):
        """Test path length computation for empty path."""
        empty_path = np.array([]).reshape(0, 6)
        length = compute_path_length(empty_path)
        assert length == 0.0
    
    def test_compute_path_length_single_point(self):
        """Test path length computation for single point."""
        single_path = np.array([[0, 0, 0, 0, 0, 0]])
        length = compute_path_length(single_path)
        assert length == 0.0
    
    def test_smooth_joint_path(self):
        """Test joint path smoothing."""
        # Create jagged path
        jagged_path = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        smoothed_path = smooth_joint_path(jagged_path, smoothing_factor=0.5, iterations=5)
        
        assert smoothed_path.shape == jagged_path.shape
        
        # Check that smoothing reduces variation
        original_variation = np.var(jagged_path[:, 0])
        smoothed_variation = np.var(smoothed_path[:, 0])
        assert smoothed_variation < original_variation
    
    def test_smooth_joint_path_short(self):
        """Test smoothing of short paths."""
        # Path with less than 3 points should be unchanged
        short_path = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1]
        ])
        
        smoothed_path = smooth_joint_path(short_path)
        assert np.array_equal(smoothed_path, short_path)


if __name__ == "__main__":
    pytest.main([__file__])

