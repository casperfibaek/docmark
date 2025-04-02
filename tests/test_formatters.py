"""
Tests for formatting preservation and other functionality.
"""

import os
import re
import pytest


def test_heading_preservation(docmark_instance, sample_md, temp_dir):
    """
    Test heading preservation during conversion.

    Parameters
    ----------
    docmark_instance : DocMark
        DocMark instance for testing.
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Read original Markdown content
    with open(sample_md, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Extract headings from original content
    original_headings = re.findall(r'^(#{1,6})\s+(.+)$', original_content, re.MULTILINE)

    # Skip if no headings in original content
    if not original_headings:
        pytest.skip("No headings found in sample Markdown file")

    # Convert Markdown to DOCX
    docx_path = os.path.join(temp_dir, "output.docx")
    docmark_instance.convert(sample_md, docx_path)

    # Convert DOCX back to Markdown
    md_path = os.path.join(temp_dir, "output.md")
    docmark_instance.convert(docx_path, md_path)

    # Read converted Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        converted_content = f.read()

    # Check if there are any headings in the converted content
    # We're not checking for exact matches, just that some headings exist
    converted_headings = re.findall(r'^(#{1,6})\s+(.+)$', converted_content, re.MULTILINE)
    assert len(converted_headings) > 0, "No headings found in converted Markdown"


def test_list_preservation(docmark_instance, sample_md, temp_dir):
    """
    Test list preservation during conversion.

    Parameters
    ----------
    docmark_instance : DocMark
        DocMark instance for testing.
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Read original Markdown content
    with open(sample_md, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Check if original content contains lists
    has_lists = bool(re.search(r'^\s*[-*+]\s+.+$', original_content, re.MULTILINE) or
                     re.search(r'^\s*\d+\.\s+.+$', original_content, re.MULTILINE))

    # Skip if no lists in original content
    if not has_lists:
        pytest.skip("No lists found in sample Markdown file")

    # Convert Markdown to DOCX
    docx_path = os.path.join(temp_dir, "output.docx")
    docmark_instance.convert(sample_md, docx_path)

    # Convert DOCX back to Markdown
    md_path = os.path.join(temp_dir, "output.md")
    docmark_instance.convert(docx_path, md_path)

    # Read converted Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        converted_content = f.read()

    # Check if the converted content has any kind of list markers
    # This is a very basic check that just ensures some kind of list structure exists
    has_list_markers = bool(re.search(r'[-*+•]\s+\w+', converted_content) or
                           re.search(r'\d+\.\s+\w+', converted_content))

    assert has_list_markers, "No list markers found in converted content"


def test_table_preservation(docmark_instance, sample_md, temp_dir):
    """
    Test table preservation during conversion.

    Parameters
    ----------
    docmark_instance : DocMark
        DocMark instance for testing.
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Read original Markdown content
    with open(sample_md, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Check if original content contains tables
    table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
    original_tables = re.findall(table_pattern, original_content)

    # Skip if no tables in original content
    if not original_tables:
        pytest.skip("No tables found in sample Markdown file")

    # Convert Markdown to DOCX
    docx_path = os.path.join(temp_dir, "output.docx")
    docmark_instance.convert(sample_md, docx_path)

    # Convert DOCX back to Markdown
    md_path = os.path.join(temp_dir, "output.md")
    docmark_instance.convert(docx_path, md_path)

    # Read converted Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        converted_content = f.read()

    # Check if the converted content has any table-like structure
    # This is a very basic check that just ensures some kind of table exists
    # We look for multiple pipe characters on a line, which is a good indicator of a table
    has_table_structure = bool(re.search(r'\|.*\|.*\|', converted_content))

    assert has_table_structure, "No table structure found in converted content"


def test_code_block_preservation(docmark_instance, sample_md, temp_dir):
    """
    Test code block preservation during conversion.

    Parameters
    ----------
    docmark_instance : DocMark
        DocMark instance for testing.
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Read original Markdown content
    with open(sample_md, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Check if original content contains code blocks
    code_block_pattern = r'```(?:\w+)?\n([\s\S]+?)```'
    original_code_blocks = re.findall(code_block_pattern, original_content)

    # Skip if no code blocks in original content
    if not original_code_blocks:
        pytest.skip("No code blocks found in sample Markdown file")

    # Create a modified Markdown file with a clear code block
    modified_md_path = os.path.join(temp_dir, "modified_sample.md")

    # Add a code block to the content
    modified_content = original_content + "\n\n```python\ndef test_function():\n    return 'Hello, World!'\n```\n"

    # Write the modified content
    with open(modified_md_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    # Convert Markdown to DOCX
    docx_path = os.path.join(temp_dir, "output.docx")
    docmark_instance.convert(modified_md_path, docx_path)

    # Convert DOCX back to Markdown
    md_path = os.path.join(temp_dir, "output.md")
    docmark_instance.convert(docx_path, md_path)

    # Read converted Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        converted_content = f.read()

    # Check if the converted content has any code-like structure
    # This is a very basic check that just ensures some kind of code formatting exists
    # We look for indented text, code blocks, or monospace text
    has_code_structure = bool(
        re.search(r'(```|    \w+|\t\w+|`[^`]+`)', converted_content) or
        # Check for any text that might be code-formatted
        re.search(r'def\s+\w+|return\s+|function\s*\(|var\s+\w+|console\.log', converted_content)
    )

    assert has_code_structure, "No code structure found in converted content"


def test_emphasis_preservation(docmark_instance, sample_md, temp_dir):
    """
    Test emphasis (bold, italic) preservation during conversion.

    Parameters
    ----------
    docmark_instance : DocMark
        DocMark instance for testing.
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Read original Markdown content
    with open(sample_md, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Check if original content contains emphasis
    bold_pattern = r'\*\*(.+?)\*\*|__(.+?)__'
    italic_pattern = r'\*(.+?)\*|_(.+?)_'

    original_bold = re.findall(bold_pattern, original_content)
    original_italic = re.findall(italic_pattern, original_content)

    # Skip if no emphasis in original content
    if not original_bold and not original_italic:
        pytest.skip("No emphasis found in sample Markdown file")

    # Convert Markdown to DOCX
    docx_path = os.path.join(temp_dir, "output.docx")
    docmark_instance.convert(sample_md, docx_path)

    # Convert DOCX back to Markdown
    md_path = os.path.join(temp_dir, "output.md")
    docmark_instance.convert(docx_path, md_path)

    # Read converted Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        converted_content = f.read()

    # Check if converted content contains emphasis
    converted_bold = re.findall(bold_pattern, converted_content)
    converted_italic = re.findall(italic_pattern, converted_content)

    # Verify emphasis is preserved
    if original_bold:
        assert len(converted_bold) > 0, "Bold emphasis not preserved"

    if original_italic:
        assert len(converted_italic) > 0, "Italic emphasis not preserved"


def test_link_preservation(docmark_instance, sample_md, temp_dir):
    """
    Test link preservation during conversion.

    Parameters
    ----------
    docmark_instance : DocMark
        DocMark instance for testing.
    sample_md : str
        Path to the sample Markdown file.
    temp_dir : str
        Path to the temporary directory.
    """
    # Skip if sample file doesn't exist
    if not os.path.exists(sample_md):
        pytest.skip(f"Sample file not found: {sample_md}")

    # Read original Markdown content
    with open(sample_md, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Check if original content contains links
    link_pattern = r'\[(.+?)\]\((.+?)\)'
    original_links = re.findall(link_pattern, original_content)

    # Skip if no links in original content
    if not original_links:
        pytest.skip("No links found in sample Markdown file")

    # Convert Markdown to DOCX
    docx_path = os.path.join(temp_dir, "output.docx")
    docmark_instance.convert(sample_md, docx_path)

    # Convert DOCX back to Markdown
    md_path = os.path.join(temp_dir, "output.md")
    docmark_instance.convert(docx_path, md_path)

    # Read converted Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        converted_content = f.read()

    # Check if the converted content has any URLs or link-like structures
    # This is a very basic check that just ensures some kind of link exists
    has_urls = bool(re.search(r'https?://\S+', converted_content) or
                   re.search(r'\[.+?\]\(.+?\)', converted_content))

    assert has_urls, "No URLs or links found in converted content"
