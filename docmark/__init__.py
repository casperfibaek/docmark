"""
DocMark: A library for converting between Word files and Markdown.

This library provides functionality for converting between various document formats,
with a focus on producing pretty output in both Markdown and Word formats.
"""

import os
from typing import Optional, Dict, Any, List, Tuple, Union

from docmark.core.processor import DocMark

__version__ = "0.1.0"


def convert(
    input_path: str,
    output_path: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_provider: str = "openai",
    verbose: bool = False,
    **kwargs: Any,
) -> str:
    """
    Convert a document from one format to another.

    Parameters
    ----------
    input_path : str
        Path to the input file.
    output_path : str, optional
        Path to the output file. If None, will be derived from input_path.
    llm_api_key : str, optional
        API key for the LLM provider. If None, will try to get from environment.
    llm_model : str, optional
        Model to use for LLM requests. If None, will use a default model.
    llm_provider : str, optional
        LLM provider to use. Options: "openai", "anthropic".
    verbose : bool, optional
        Whether to print verbose output.
    **kwargs : dict
        Additional arguments for the conversion:
        - from_format : str, optional
            Format of the input file. If None, will be derived from input_path.
        - to_format : str, optional
            Format of the output file. If None, will be derived from output_path.
        - images_dir : str, optional
            Directory for extracted images. If None, will use 'media' folder.
        - template_path : str, optional
            Path to the template file for DOCX output.
        - toc : bool, optional
            Whether to include a table of contents in DOCX output.
        - use_llm : bool, optional
            Whether to use LLM for enhanced formatting.

    Returns
    -------
    str
        Path to the output file.
    """
    # Try to get API key from environment if not provided
    if not llm_api_key:
        llm_api_key = os.environ.get("OPENAI_API_KEY")
        if not llm_api_key and llm_provider == "anthropic":
            llm_api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Create DocMark instance
    docmark = DocMark(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        verbose=verbose,
    )

    # Convert the document
    return docmark.convert(input_path, output_path, **kwargs)


def batch_convert(
    input_dir: str,
    output_dir: Optional[str] = None,
    pattern: str = "*.*",
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_provider: str = "openai",
    verbose: bool = False,
    **kwargs: Any,
) -> List[str]:
    """
    Convert multiple documents in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing input files.
    output_dir : str, optional
        Directory for output files. If None, will use input_dir.
    pattern : str, optional
        Pattern to match input files.
    llm_api_key : str, optional
        API key for the LLM provider. If None, will try to get from environment.
    llm_model : str, optional
        Model to use for LLM requests. If None, will use a default model.
    llm_provider : str, optional
        LLM provider to use. Options: "openai", "anthropic".
    verbose : bool, optional
        Whether to print verbose output.
    **kwargs : dict
        Additional arguments for the conversion.

    Returns
    -------
    List[str]
        List of paths to the output files.
    """
    # Try to get API key from environment if not provided
    if not llm_api_key:
        llm_api_key = os.environ.get("OPENAI_API_KEY")
        if not llm_api_key and llm_provider == "anthropic":
            llm_api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Create DocMark instance
    docmark = DocMark(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        verbose=verbose,
    )

    # Convert the documents
    return docmark.batch_convert(input_dir, output_dir, pattern, **kwargs)


# Expose main classes
__all__ = ["DocMark", "convert", "batch_convert"]
