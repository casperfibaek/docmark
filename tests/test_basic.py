"""
Basic tests for the docmark package.
"""

import pytest

def test_import():
    """Test that the main docmark package can be imported."""
    try:
        import docmark
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import docmark package: {e}")

def test_convert_function_exists():
    """Test that the convert function is available."""
    try:
        from docmark import convert
        assert callable(convert)
    except ImportError as e:
        pytest.fail(f"Failed to import convert function from docmark: {e}")

def test_batch_convert_function_exists():
    """Test that the batch_convert function is available."""
    try:
        from docmark import batch_convert
        assert callable(batch_convert)
    except ImportError as e:
        pytest.fail(f"Failed to import batch_convert function from docmark: {e}")