"""
Tests for file utility functions in docmark.utils.file.
"""

import os
import pytest
from docmark.utils import file as file_utils

def test_get_file_type():
    """Test the get_file_type function."""
    assert file_utils.get_file_type("document.docx") == "docx"
    assert file_utils.get_file_type("document.md") == "md"
    assert file_utils.get_file_type("document.markdown") == "md"
    assert file_utils.get_file_type("document.pdf") == "pdf"
    assert file_utils.get_file_type("document.html") == "html"
    assert file_utils.get_file_type("document.htm") == "html"
    assert file_utils.get_file_type("document.txt") == "unknown" # Assuming txt is unknown for conversion
    assert file_utils.get_file_type("document") == "unknown"
    assert file_utils.get_file_type("archive.zip") == "unknown"

def test_get_output_path_no_output_path():
    """Test get_output_path when output_path is not provided."""
    assert file_utils.get_output_path("input.docx") == "input.md"
    assert file_utils.get_output_path("input.md") == "input.docx"
    assert file_utils.get_output_path("input.pdf") == "input.md"
    assert file_utils.get_output_path("input.html") == "input.md"
    assert file_utils.get_output_path("input.txt") == "input.txt" # Fallback for unknown
    assert file_utils.get_output_path("path/to/input.docx") == "path/to/input.md"

def test_get_output_path_with_output_format():
    """Test get_output_path when output_format is provided."""
    assert file_utils.get_output_path("input.docx", output_format="md") == "input.md"
    assert file_utils.get_output_path("input.md", output_format="docx") == "input.docx"
    assert file_utils.get_output_path("input.pdf", output_format="md") == "input.md"
    assert file_utils.get_output_path("input.html", output_format="md") == "input.md"
    assert file_utils.get_output_path("input.txt", output_format="docx") == "input.docx"
    assert file_utils.get_output_path("path/to/input.docx", output_format="md") == "path/to/input.md"

def test_get_output_path_with_output_path():
    """Test get_output_path when output_path is provided."""
    assert file_utils.get_output_path("input.docx", output_path="output.md") == "output.md"
    assert file_utils.get_output_path("input.md", output_path="new_doc.docx") == "new_doc.docx"
    assert file_utils.get_output_path("input.pdf", output_path="result.md") == "result.md"
    assert file_utils.get_output_path("input.html", output_path="final/output.md") == "final/output.md"
    # output_path overrides output_format if both are given
    assert file_utils.get_output_path("input.docx", output_path="output.md", output_format="docx") == "output.md"

def test_find_files(temp_dir):
    """Test the find_files function."""
    # Create some test files
    open(os.path.join(temp_dir, "file1.md"), "w").close()
    open(os.path.join(temp_dir, "file2.md"), "w").close()
    open(os.path.join(temp_dir, "file3.txt"), "w").close()
    os.makedirs(os.path.join(temp_dir, "subdir"), exist_ok=True)
    open(os.path.join(temp_dir, "subdir", "file4.md"), "w").close()

    # Test finding markdown files
    md_files = file_utils.find_files(temp_dir, "*.md")
    assert len(md_files) == 2
    assert os.path.join(temp_dir, "file1.md") in md_files
    assert os.path.join(temp_dir, "file2.md") in md_files

    # Test finding text files
    txt_files = file_utils.find_files(temp_dir, "*.txt")
    assert len(txt_files) == 1
    assert os.path.join(temp_dir, "file3.txt") in txt_files

    # Test finding all files
    all_files = file_utils.find_files(temp_dir, "*.*")
    assert len(all_files) == 3 # Should not be recursive by default

    # Test finding in non-existent directory
    assert file_utils.find_files("non_existent_dir", "*.*") == []