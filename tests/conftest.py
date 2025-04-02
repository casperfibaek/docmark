"""
Pytest configuration and fixtures for DocMark tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator, Tuple

import pytest

from docmark import DocMark


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for test files.
    
    Returns
    -------
    Generator[str, None, None]
        Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def temp_media_dir(temp_dir: str) -> str:
    """
    Create a temporary media directory for test images.
    
    Parameters
    ----------
    temp_dir : str
        Path to the temporary directory.
        
    Returns
    -------
    str
        Path to the temporary media directory.
    """
    media_dir = os.path.join(temp_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    return media_dir


@pytest.fixture
def docmark_instance(temp_media_dir: str) -> DocMark:
    """
    Create a DocMark instance for testing.
    
    Parameters
    ----------
    temp_media_dir : str
        Path to the temporary media directory.
        
    Returns
    -------
    DocMark
        DocMark instance for testing.
    """
    return DocMark(verbose=True)


@pytest.fixture
def test_data_dir() -> str:
    """
    Get the path to the test data directory.
    
    Returns
    -------
    str
        Path to the test data directory.
    """
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def sample_docx(test_data_dir: str) -> str:
    """
    Get the path to the sample DOCX file.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
        
    Returns
    -------
    str
        Path to the sample DOCX file.
    """
    return os.path.join(test_data_dir, "sample.docx")


@pytest.fixture
def sample_md(test_data_dir: str) -> str:
    """
    Get the path to the sample Markdown file.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
        
    Returns
    -------
    str
        Path to the sample Markdown file.
    """
    return os.path.join(test_data_dir, "sample.md")


@pytest.fixture
def sample_html(test_data_dir: str) -> str:
    """
    Get the path to the sample HTML file.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
        
    Returns
    -------
    str
        Path to the sample HTML file.
    """
    return os.path.join(test_data_dir, "sample.html")


@pytest.fixture
def sample_with_images_docx(test_data_dir: str) -> str:
    """
    Get the path to the sample DOCX file with images.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
        
    Returns
    -------
    str
        Path to the sample DOCX file with images.
    """
    return os.path.join(test_data_dir, "sample_with_images.docx")


@pytest.fixture
def sample_with_images_md(test_data_dir: str) -> str:
    """
    Get the path to the sample Markdown file with images.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
        
    Returns
    -------
    str
        Path to the sample Markdown file with images.
    """
    return os.path.join(test_data_dir, "sample_with_images.md")


@pytest.fixture
def sample_with_images_html(test_data_dir: str) -> str:
    """
    Get the path to the sample HTML file with images.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
        
    Returns
    -------
    str
        Path to the sample HTML file with images.
    """
    return os.path.join(test_data_dir, "sample_with_images.html")


@pytest.fixture
def copy_template_docx(test_data_dir: str) -> None:
    """
    Copy the template DOCX file to the test data directory.
    
    Parameters
    ----------
    test_data_dir : str
        Path to the test data directory.
    """
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                "docmark", "templates", "default.docx")
    if os.path.exists(template_path):
        shutil.copy(template_path, os.path.join(test_data_dir, "template.docx"))
