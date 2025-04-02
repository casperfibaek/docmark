"""
Tests for the command-line interface.
"""

import os
import pytest


def test_cli_help():
    """Test the CLI help command."""
    # Run the CLI help command directly using the Python API
    from docmark.cli import cli
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    # Verify the command succeeded
    assert result.exit_code == 0

    # Verify the output contains expected help text
    assert "DocMark: Convert between Word files and Markdown" in result.output
    assert "Commands:" in result.output
    assert "convert" in result.output


def test_cli_version():
    """Test the CLI version command."""
    # Run the CLI version command directly using the Python API
    from docmark.cli import cli
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    # Verify the command succeeded
    assert result.exit_code == 0

    # Verify the output contains a version number
    assert "version" in result.output


def test_cli_convert_command(sample_docx, temp_dir):
    """
    Test the CLI convert command.

    Parameters
    ----------
    sample_docx : str
        Path to the sample DOCX file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_docx):
        pytest.skip(f"Sample file not found: {sample_docx}")

    # Output path
    output_path = os.path.join(temp_dir, "output.md")

    # Run the CLI convert command directly using the Python API
    from docmark.cli import cli
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli, ["convert", sample_docx, output_path])

    # Verify the command succeeded
    assert result.exit_code == 0

    # Verify the output file exists
    assert os.path.exists(output_path)

    # Verify the output file is not empty
    assert os.path.getsize(output_path) > 0


def test_cli_batch_command(sample_docx, temp_dir):
    """
    Test the CLI batch command.

    Parameters
    ----------
    sample_docx : str
        Path to the sample DOCX file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_docx):
        pytest.skip(f"Sample file not found: {sample_docx}")

    # Create a test directory with multiple sample files
    test_batch_dir = os.path.join(temp_dir, "batch_test_dir")
    os.makedirs(test_batch_dir, exist_ok=True)

    # Copy the sample file to the test directory with multiple names
    import shutil
    for i in range(1, 3):
        test_file = os.path.join(test_batch_dir, f"test_sample_{i}.docx")
        shutil.copy(sample_docx, test_file)

    # Create a temporary output directory
    output_dir = os.path.join(temp_dir, "batch_output")
    os.makedirs(output_dir, exist_ok=True)

    # Run the batch command directly using the DocMark API
    from docmark import DocMark
    docmark = DocMark(verbose=True)
    output_files = docmark.batch_convert(
        test_batch_dir,
        output_dir,
        pattern="*.docx"
    )

    # Verify output files were created
    assert len(output_files) > 0, "No output files created"

    # Verify each output file exists
    for output_file in output_files:
        assert os.path.exists(output_file), f"Output file not found: {output_file}"
        assert os.path.getsize(output_file) > 0, f"Output file is empty: {output_file}"


def test_cli_convert_with_options(sample_md, temp_dir):
    """
    Test the CLI convert command with options.

    Parameters
    ----------
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Output path
    output_path = os.path.join(temp_dir, "output.docx")

    # Run the CLI convert command with options directly using the Python API
    from docmark.cli import cli
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli, [
        "convert",
        sample_md,
        output_path,
        "--toc",  # Include table of contents
    ])

    # Verify the command succeeded
    assert result.exit_code == 0

    # Verify the output file exists
    assert os.path.exists(output_path)

    # Verify the output file is not empty
    assert os.path.getsize(output_path) > 0
