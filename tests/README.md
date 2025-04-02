# DocMark Test Framework

This directory contains the test framework for DocMark, a library for converting between different document formats.

## Test Structure

The test framework is organized as follows:

- `conftest.py`: Contains pytest fixtures and configuration
- `test_bidirectional.py`: Tests for bidirectional conversion between formats
- `test_image_handling.py`: Tests for image extraction and naming
- `test_formatters.py`: Tests for formatting preservation
- `test_cli.py`: Tests for the command-line interface
- `test_data/`: Directory containing sample files for testing

## Running Tests

To run the tests, use pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bidirectional.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=docmark
```

## Test Data

The `test_data` directory contains sample files for testing:

- `sample.md`: Sample Markdown file
- `sample.html`: Sample HTML file
- `sample.docx`: Sample DOCX file
- `sample_with_images.md`: Sample Markdown file with images
- `sample_with_images.html`: Sample HTML file with images
- `sample_with_images.docx`: Sample DOCX file with images

## Image Naming Convention

The test framework verifies that images extracted from documents follow the naming convention:

```
{document_filename}_img{sequential_number}.{extension}
```

For example, images extracted from `sample.docx` would be named:
- `sample_img1.png`
- `sample_img2.png`
- etc.

## Bidirectional Conversion

The test framework verifies that bidirectional conversion works correctly:

- DOCX → MD → DOCX
- MD → DOCX → MD
- HTML → MD → HTML (partial, as HTML → HTML is not directly supported)

## Formatting Preservation

The test framework verifies that formatting is preserved during conversion:

- Headings
- Lists (ordered and unordered)
- Tables
- Code blocks
- Emphasis (bold, italic)
- Links

## Adding New Tests

To add new tests:

1. Create a new test file in the `tests` directory
2. Import the necessary modules and fixtures
3. Write test functions using pytest assertions
4. Add sample files to the `test_data` directory if needed
