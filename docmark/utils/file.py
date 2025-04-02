"""
File utility module for DocMark.

This module provides functionality for file operations, including
path handling, file detection, and I/O operations.
"""

import os
import shutil
import tempfile
import mimetypes
from typing import Optional, List, Dict, Any


def ensure_dir_exists(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    directory : str
        Path to the directory.

    Returns
    -------
    str
        Path to the directory.
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def get_file_type(file_path: str) -> str:
    """
    Get the type of a file based on its extension.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    str
        File type (e.g., 'docx', 'md', 'pdf', 'html').
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.docx':
        return 'docx'
    if ext in ('.md', '.markdown'):
        return 'md'
    if ext == '.pdf':
        return 'pdf'
    if ext in ('.html', '.htm'):
        return 'html'

    # Try to guess based on MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return 'docx'
        if mime_type == 'text/markdown':
            return 'md'
        if mime_type == 'application/pdf':
            return 'pdf'
        if mime_type == 'text/html':
            return 'html'

    # Default to unknown
    return 'unknown'


def get_output_path(input_path: str, output_path: Optional[str] = None,
                    output_format: Optional[str] = None) -> str:
    """
    Determine the output path for a conversion.

    Parameters
    ----------
    input_path : str
        Path to the input file.
    output_path : str, optional
        Path to the output file. If None, will be derived from input_path.
    output_format : str, optional
        Format of the output file. If None, will be derived from output_path.

    Returns
    -------
    str
        Path to the output file.
    """
    if output_path:
        return output_path

    # Derive output path from input path
    input_dir = os.path.dirname(input_path)
    input_name = os.path.splitext(os.path.basename(input_path))[0]

    if output_format:
        if output_format == 'docx':
            ext = '.docx'
        elif output_format == 'md':
            ext = '.md'
        elif output_format == 'pdf':
            ext = '.pdf'
        elif output_format == 'html':
            ext = '.html'
        else:
            ext = '.' + output_format
    else:
        # Try to guess from input path
        input_type = get_file_type(input_path)
        if input_type == 'docx':
            ext = '.md'  # Default conversion: docx -> md
        elif input_type == 'md':
            ext = '.docx'  # Default conversion: md -> docx
        elif input_type == 'pdf':
            ext = '.md'  # Default conversion: pdf -> md
        elif input_type == 'html':
            ext = '.md'  # Default conversion: html -> md
        else:
            ext = '.txt'  # Fallback

    # Construct the path using os.path.join
    output_path_os = os.path.join(input_dir, f"{input_name}{ext}")

    # Normalize separators to forward slashes for consistency
    return output_path_os.replace("\\", "/")


def create_temp_file(content: str, suffix: str = '.txt') -> str:
    """
    Create a temporary file with the given content.

    Parameters
    ----------
    content : str
        Content to write to the file.
    suffix : str, optional
        Suffix for the temporary file.

    Returns
    -------
    str
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(content)
    return temp_file.name


def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    str
        Contents of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    """
    Write content to a file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    content : str
        Content to write to the file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def copy_file(src: str, dst: str) -> str:
    """
    Copy a file from source to destination.

    Parameters
    ----------
    src : str
        Path to the source file.
    dst : str
        Path to the destination file.

    Returns
    -------
    str
        Path to the destination file.
    """
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    return shutil.copy2(src, dst)


def get_resource_path(resource_name: str) -> str:
    """
    Get the path to a resource file.

    Parameters
    ----------
    resource_name : str
        Name of the resource.

    Returns
    -------
    str
        Path to the resource file.
    """
    # Get the path to the docmark package
    package_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the path to the resource
    resource_path: str = os.path.join(package_dir, resource_name)

    if not os.path.exists(resource_path):
        raise FileNotFoundError(f"Resource not found: {resource_name}")

    return resource_path


def get_template_path(template_name: Optional[str] = None) -> str:
    """
    Get the path to a template file.

    Parameters
    ----------
    template_name : str, optional
        Name of the template. If None, returns the default template.

    Returns
    -------
    str
        Path to the template file.
    """
    if template_name:
        # If an absolute path is provided, use it directly
        if os.path.isabs(template_name) and os.path.exists(template_name):
            return template_name

        # Try to find the template in the templates directory
        try:
            return get_resource_path(os.path.join('templates', template_name))
        except FileNotFoundError as exc:
            # If not found, check if the template exists as a standalone file
            if os.path.exists(template_name):
                return template_name
            raise FileNotFoundError(f"Template not found: {template_name}") from exc
    else:
        # Return the default template
        return get_resource_path(os.path.join('templates', 'default.docx'))


def find_files(directory: str, pattern: str) -> List[str]:
    """
    Find files in a directory matching a pattern.

    Parameters
    ----------
    directory : str
        Directory to search in.
    pattern : str
        Pattern to match (e.g., '*.md').

    Returns
    -------
    List[str]
        List of matching file paths.
    """
    import glob

    # Ensure directory exists
    if not os.path.exists(directory):
        return []

    # Construct the pattern
    search_pattern = os.path.join(directory, pattern)

    # Find matching files
    return glob.glob(search_pattern)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing file information.
    """
    if not os.path.exists(file_path):
        return {}

    file_stat = os.stat(file_path)

    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'extension': os.path.splitext(file_path)[1],
        'size': file_stat.st_size,
        'modified': file_stat.st_mtime,
        'type': get_file_type(file_path),
    }
