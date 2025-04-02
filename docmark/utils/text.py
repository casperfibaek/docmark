"""
Text utility module for DocMark.

This module provides functionality for text processing, including
formatting, cleaning, and manipulation.
"""

import re
import os
from typing import List, Dict, Any
import subprocess
import tempfile


def ensure_atx_headings(markdown_content: str) -> str:
    """
    Ensure headings are properly formatted in ATX style (with # symbols).

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with properly formatted ATX headings.
    """
    # Fix potential setext-style headings (=== and --- underlines)
    # Match a line of text followed by a line of === or ---
    setext_h1_pattern = r'([^\n]+)\n=+\s*\n'
    setext_h2_pattern = r'([^\n]+)\n-+\s*\n'

    # Replace with ATX style
    markdown_content = re.sub(setext_h1_pattern, r'# \1\n\n', markdown_content)
    markdown_content = re.sub(setext_h2_pattern, r'## \1\n\n', markdown_content)

    # Ensure proper spacing around ATX headings
    heading_pattern = r'(^|\n)(#{1,6})[ \t]+(.+?)[ \t]*(?:\n|$)'
    markdown_content = re.sub(heading_pattern, r'\1\2 \3\n\n', markdown_content)

    return markdown_content


def fix_list_formatting(markdown_content: str) -> str:
    """
    Fix list formatting in markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with properly formatted lists.
    """
    # Ensure proper spacing before lists
    list_start_pattern = r'(^|\n)([^\n]+\n)([*\-+]|\d+\.)[ \t]+'
    markdown_content = re.sub(list_start_pattern, r'\1\2\n\3 ', markdown_content)

    # Ensure proper indentation for nested lists
    lines = markdown_content.split('\n')
    result_lines = []
    in_list = False
    list_indent = 0

    for i, line in enumerate(lines):
        # Check if this line is a list item
        list_match = re.match(r'^(\s*)([*\-+]|\d+\.)(\s+)(.*)$', line)

        if list_match:
            indent, marker, space, content = list_match.groups()

            # If this is the start of a new list
            if not in_list:
                in_list = True
                list_indent = len(indent)
                result_lines.append(line)
            else:
                # If this is a nested list item
                if len(indent) > list_indent:
                    # Ensure consistent indentation (4 spaces per level)
                    level = (len(indent) - list_indent) // 4 + 1
                    proper_indent = ' ' * (list_indent + (level - 1) * 4)
                    result_lines.append(f"{proper_indent}{marker}{space}{content}")
                else:
                    # Same level or back to a higher level
                    list_indent = len(indent)
                    result_lines.append(line)
        else:
            # If line is empty or not a list item
            if line.strip() == '':
                # Empty line might end a list
                if i + 1 < len(lines) and not re.match(r'^\s*([*\-+]|\d+\.)\s+', lines[i + 1]):
                    in_list = False

            result_lines.append(line)

    return '\n'.join(result_lines)


def fix_code_blocks(markdown_content: str) -> str:
    """
    Fix code block formatting in markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with properly formatted code blocks.
    """
    # Replace indented code blocks with fenced code blocks
    lines = markdown_content.split('\n')
    result_lines = []
    in_code_block = False
    code_block_indent = 0
    code_block_lines = []

    for i, line in enumerate(lines):
        # Check if this line is indented by 4 or more spaces (or a tab)
        indent_match = re.match(r'^(\s{4,}|\t)(.*)$', line)

        if indent_match and not in_code_block:
            # Start of a potential code block
            prev_line_empty = (i == 0) or (lines[i-1].strip() == '')

            if prev_line_empty:
                in_code_block = True
                code_block_indent = len(indent_match.group(1))
                code_block_lines = [indent_match.group(2)]
                result_lines.append('```')
            else:
                result_lines.append(line)
        elif in_code_block:
            # Check if this line is still part of the code block
            if line.strip() == '':
                # Empty line might be part of the code block
                code_block_lines.append('')
            elif line.startswith(' ' * code_block_indent) or line.startswith('\t'):
                # Still indented, part of the code block
                if line.startswith(' ' * code_block_indent):
                    code_block_lines.append(line[code_block_indent:])
                else:
                    code_block_lines.append(line[1:])  # Remove tab
            else:
                # End of code block
                in_code_block = False
                result_lines.extend(code_block_lines)
                result_lines.append('```')
                result_lines.append(line)
        else:
            result_lines.append(line)

    # Close any open code block
    if in_code_block:
        result_lines.extend(code_block_lines)
        result_lines.append('```')

    # Fix fenced code blocks (ensure proper syntax)
    markdown_content = '\n'.join(result_lines)

    # Ensure language is specified in fenced code blocks
    fenced_block_pattern = r'```\s*(\w*)\n'

    def replace_fence(match: re.Match) -> str:
        lang = match.group(1).strip()
        if not lang:
            # Try to guess language from content or default to text
            return '```text\n'
        return f'```{lang}\n'

    markdown_content = re.sub(fenced_block_pattern, replace_fence, markdown_content)

    return markdown_content


def fix_table_formatting(markdown_content: str) -> str:
    """
    Fix table formatting in markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with properly formatted tables.
    """
    # Find all tables in the markdown content
    table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'

    def fix_table(match: re.Match) -> str:
        table = match.group(0)
        lines = table.split('\n')

        if not lines:
            return table

        # Get the number of columns from the header row
        header = lines[0]
        columns = header.count('|') - 1

        # Fix header row
        header_cells = [cell.strip() for cell in header.split('|')[1:-1]]
        header_cells = [cell if cell else ' ' for cell in header_cells]  # Replace empty cells with space

        # Ensure header has the right number of columns
        while len(header_cells) < columns:
            header_cells.append(' ')

        # Fix separator row
        separator = lines[1]
        separator_cells = separator.split('|')[1:-1]

        # Create proper separator cells with alignment
        fixed_separator_cells = []
        for cell in separator_cells:
            cell = cell.strip()
            if cell.startswith(':') and cell.endswith(':'):
                fixed_separator_cells.append(':---:')  # Center align
            elif cell.startswith(':'):
                fixed_separator_cells.append(':---')   # Left align
            elif cell.endswith(':'):
                fixed_separator_cells.append('---:')   # Right align
            else:
                fixed_separator_cells.append('---')    # Default align

        # Ensure separator has the right number of columns
        while len(fixed_separator_cells) < columns:
            fixed_separator_cells.append('---')

        # Fix data rows
        fixed_data_rows = []
        for i in range(2, len(lines)):
            if not lines[i]:
                continue

            data_cells = [cell.strip() for cell in lines[i].split('|')[1:-1]]

            # Ensure data row has the right number of columns
            while len(data_cells) < columns:
                data_cells.append(' ')

            fixed_data_rows.append('| ' + ' | '.join(data_cells) + ' |')

        # Reconstruct the table
        fixed_table = '| ' + ' | '.join(header_cells) + ' |\n'
        fixed_table += '| ' + ' | '.join(fixed_separator_cells) + ' |\n'
        fixed_table += '\n'.join(fixed_data_rows)

        return fixed_table

    result: str = re.sub(table_pattern, fix_table, markdown_content)
    return result


def fix_image_links(markdown_content: str) -> str:
    """
    Fix image and link syntax in markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with properly formatted images and links.
    """
    # Fix image syntax
    image_pattern = r'!\[(.*?)\]\s*\((.*?)\s*(?:"(.*?)")?\)'

    def fix_image(match: re.Match) -> str:
        alt_text = match.group(1) or ''
        url = match.group(2)
        title = match.group(3) or ''

        if title:
            return f'![{alt_text}]({url} "{title}")'
        return f'![{alt_text}]({url})'

    markdown_content = re.sub(image_pattern, fix_image, markdown_content)

    # Fix link syntax
    link_pattern = r'\[(.*?)\]\s*\((.*?)\s*(?:"(.*?)")?\)'

    def fix_link(match: re.Match) -> str:
        text = match.group(1) or ''
        url = match.group(2)
        title = match.group(3) or ''

        if title:
            return f'[{text}]({url} "{title}")'
        return f'[{text}]({url})'

    markdown_content = re.sub(link_pattern, fix_link, markdown_content)

    return markdown_content


def ensure_proper_spacing(markdown_content: str) -> str:
    """
    Ensure proper spacing in markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with proper spacing.
    """
    # Ensure single blank line before headings
    heading_pattern = r'([^\n])\n(#{1,6} )'
    markdown_content = re.sub(heading_pattern, r'\1\n\n\2', markdown_content)

    # Ensure single blank line after headings
    after_heading_pattern = r'(#{1,6} .+)\n([^\n])'
    markdown_content = re.sub(after_heading_pattern, r'\1\n\n\2', markdown_content)

    # Ensure single blank line before lists
    list_pattern = r'([^\n])\n([*\-+]|\d+\.) '
    markdown_content = re.sub(list_pattern, r'\1\n\n\2 ', markdown_content)

    # Ensure single blank line before code blocks
    code_pattern = r'([^\n])\n```'
    markdown_content = re.sub(code_pattern, r'\1\n\n```', markdown_content)

    # Ensure single blank line after code blocks
    after_code_pattern = r'```\n([^\n])'
    markdown_content = re.sub(after_code_pattern, r'```\n\n\1', markdown_content)

    # Remove more than two consecutive blank lines
    multi_blank_pattern = r'\n{3,}'
    markdown_content = re.sub(multi_blank_pattern, r'\n\n', markdown_content)

    return markdown_content


def markdown_linter(markdown_content: str) -> str:
    """
    Apply pymarkdown fix to the markdown content using a temporary file.

    Parameters
    ----------
    markdown_content : str
        The markdown content to lint.

    Returns
    -------
    str
        The linted markdown content.
    """
    temp_file = None
    temp_file_path = None # Define outside try block for finally
    try:
        # Create a temporary file with .md extension
        # delete=False is important because pymarkdown needs the path after the file is closed
        temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False, encoding='utf-8')
        temp_file_path = temp_file.name
        temp_file.write(markdown_content)
        temp_file.close() # Close the file so pymarkdown can access it (especially on Windows)

        # Run pymarkdown fix command
        try:
            # Use capture_output=True and text=True to handle potential output/errors
            # check=True will raise CalledProcessError if pymarkdown returns non-zero exit code
            subprocess.run(['pymarkdown', 'fix', temp_file_path], check=True, capture_output=True, text=True, encoding='utf-8')

            # If pymarkdown succeeded, read the modified content back
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

        except FileNotFoundError:
            # Handle case where pymarkdown is not installed or not in PATH
            print(f"Warning: 'pymarkdown' command not found. Skipping pymarkdown fix step.")
            # Keep the content as is
        except subprocess.CalledProcessError as e:
            # Handle case where pymarkdown command fails
            print(f"Warning: 'pymarkdown fix' failed with error: {e}. Stderr: {e.stderr}. Skipping pymarkdown fix step.")
            # Keep the content as is

    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")

    return markdown_content


def fix_markdown_formatting(markdown_content: str) -> str:
    """
    Fix markdown formatting issues.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Properly formatted markdown content.
    """
    # Apply external linter
    markdown_content = markdown_linter(markdown_content)

    # Apply all formatting fixes
    markdown_content = ensure_atx_headings(markdown_content)
    markdown_content = fix_list_formatting(markdown_content)
    markdown_content = fix_code_blocks(markdown_content)
    markdown_content = fix_table_formatting(markdown_content)
    markdown_content = fix_image_links(markdown_content)
    markdown_content = ensure_proper_spacing(markdown_content)

    return markdown_content


def extract_toc_entries(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extract table of contents entries from markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    List[Dict[str, Any]]
        List of TOC entries, each with 'text', 'level', and 'anchor' keys.
    """
    # Find all headings
    heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s+\{#([a-zA-Z0-9_-]+)\})?\s*$'

    entries = []
    for line in markdown_content.split('\n'):
        match = re.match(heading_pattern, line)
        if match:
            hashes, text, anchor = match.groups()
            level = len(hashes)

            # Generate anchor if not provided
            if not anchor:
                anchor = text.lower()
                anchor = re.sub(r'[^\w\- ]', '', anchor)  # Remove special chars
                anchor = re.sub(r'\s+', '-', anchor)      # Replace spaces with hyphens

            entries.append({
                'text': text,
                'level': level,
                'anchor': anchor
            })

    return entries


def generate_toc(markdown_content: str, max_level: int = 3) -> str:
    """
    Generate a table of contents from markdown content.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.
    max_level : int, optional
        Maximum heading level to include in TOC.

    Returns
    -------
    str
        Markdown table of contents.
    """
    entries = extract_toc_entries(markdown_content)

    toc_lines = ["# Table of Contents", ""]

    for entry in entries:
        if entry['level'] <= max_level:
            indent = "  " * (entry['level'] - 1)
            toc_lines.append(f"{indent}- [{entry['text']}](#{entry['anchor']})")

    return '\n'.join(toc_lines)


def replace_toc_marker(markdown_content: str) -> str:
    """
    Replace TOC marker with generated TOC.

    Parameters
    ----------
    markdown_content : str
        The markdown content to process.

    Returns
    -------
    str
        Markdown content with TOC marker replaced with generated TOC.
    """
    toc_marker_pattern = r'\[TOC\]|\[\[TOC\]\]'

    if re.search(toc_marker_pattern, markdown_content):
        toc = generate_toc(markdown_content)
        return re.sub(toc_marker_pattern, toc, markdown_content)
    return markdown_content


def clean_html(html_content: str) -> str:
    """
    Clean HTML content for better markdown conversion.

    Parameters
    ----------
    html_content : str
        The HTML content to clean.

    Returns
    -------
    str
        Cleaned HTML content.
    """
    # Remove unnecessary whitespace
    html_content = re.sub(r'\s+', ' ', html_content)

    # Fix common issues with HTML that can cause problems in conversion

    # Ensure proper nesting of list items
    html_content = re.sub(r'</(ul|ol)>\s*<li>', r'</\1><ul><li>', html_content)
    html_content = re.sub(r'</li>\s*<(ul|ol)>', r'</li><\1>', html_content)

    # Ensure proper paragraph breaks
    html_content = re.sub(r'<br>\s*<br>', r'</p><p>', html_content)

    # Ensure proper heading structure
    for i in range(6, 0, -1):
        html_content = re.sub(f'<h{i}>(.*?)</h{i}>', f'<h{i}>\\1</h{i}>', html_content)

    return html_content


def html_to_plain_text(html_content: str) -> str:
    """
    Convert HTML to plain text.

    Parameters
    ----------
    html_content : str
        The HTML content to convert.

    Returns
    -------
    str
        Plain text content.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', html_content)

    # Decode HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
