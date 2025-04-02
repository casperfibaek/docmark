"""
Processor module for DocMark.

This module provides the core functionality for converting between
different document formats.
"""

import os
from typing import Optional, Any, List

from docmark.core.llm import LLMManager
from docmark.core.formatter import MarkdownFormatter, DocxFormatter
from docmark.utils.image import ImageProcessor
from docmark.utils.file import get_file_type, get_output_path, get_template_path
from docmark.utils.file import find_files

class Processor:
    """Base processor class for document conversion."""

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: str = "openai",
        verbose: bool = False,
    ):
        """
        Initialize the processor.

        Parameters
        ----------
        llm_api_key : str, optional
            API key for the LLM provider.
        llm_model : str, optional
            Model to use for LLM requests.
        llm_provider : str, optional
            LLM provider to use.
        verbose : bool, optional
            Whether to print verbose output.
        """
        self.verbose = verbose

        # Initialize LLM manager if API key is provided
        self.llm_manager = None
        if llm_api_key:
            self.llm_manager = LLMManager(
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
            )
            try:
                self.llm_manager.initialize()
                if self.verbose:
                    print(f"Initialized LLM manager with provider: {llm_provider}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to initialize LLM manager: {str(e)}")
                self.llm_manager = None

        # Initialize formatters
        self.markdown_formatter = MarkdownFormatter(llm_manager=self.llm_manager)

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
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
        **kwargs : dict
            Additional arguments for the conversion.

        Returns
        -------
        str
            Path to the output file.
        """
        # Determine input and output formats
        input_format = kwargs.get('from_format') or get_file_type(input_path)
        output_format = kwargs.get('to_format')

        # Determine output path if not provided
        if not output_path:
            output_path = get_output_path(input_path, output_format=output_format)
        else:
            # If output path is provided but not output format, derive from output path
            if not output_format:
                output_format = get_file_type(output_path)

        if self.verbose:
            print(f"Converting {input_path} ({input_format}) to {output_path} ({output_format})")

        # Perform the conversion based on input and output formats
        if input_format == 'docx' and output_format == 'md':
            return self.convert_docx_to_markdown(input_path, output_path, **kwargs)
        elif input_format == 'md' and output_format == 'docx':
            return self.convert_markdown_to_docx(input_path, output_path, **kwargs)
        elif input_format == 'pdf' and output_format == 'md':
            return self.convert_pdf_to_markdown(input_path, output_path, **kwargs)
        elif input_format == 'html' and output_format == 'md':
            return self.convert_html_to_markdown(input_path, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported conversion: {input_format} to {output_format}")

    def convert_docx_to_markdown(
        self,
        docx_path: str,
        output_path: str,
        **kwargs: Any,
    ) -> str:
        """
        Convert a DOCX file to Markdown.

        Parameters
        ----------
        docx_path : str
            Path to the DOCX file.
        output_path : str
            Path to the output Markdown file.
        **kwargs : dict
            Additional arguments for the conversion.

        Returns
        -------
        str
            Path to the output Markdown file.
        """
        # Import here to avoid circular imports
        from docmark.converters.docx import DocxToMarkdownConverter

        # Get images directory
        images_dir = kwargs.get('images_dir')
        if not images_dir:
            # Use 'media' folder in same directory as output file
            images_dir = os.path.join(os.path.dirname(output_path), "media")

        # Create converter
        converter = DocxToMarkdownConverter(
            llm_manager=self.llm_manager,
            image_processor=ImageProcessor(images_dir=images_dir),
            verbose=self.verbose,
        )

        # Convert the document
        markdown_content = converter.convert(docx_path)

        # Format the markdown content
        markdown_content = self.markdown_formatter.format(
            markdown_content,
            use_llm=kwargs.get('use_llm', True),
        )

        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        if self.verbose:
            print(f"Saved Markdown to {output_path}")

        return output_path

    def convert_markdown_to_docx(
        self,
        markdown_path: str,
        output_path: str,
        **kwargs: Any,
    ) -> str:
        """
        Convert a Markdown file to DOCX.

        Parameters
        ----------
        markdown_path : str
            Path to the Markdown file.
        output_path : str
            Path to the output DOCX file.
        **kwargs : dict
            Additional arguments for the conversion.

        Returns
        -------
        str
            Path to the output DOCX file.
        """
        # Import here to avoid circular imports
        from docmark.converters.markdown import MarkdownToDocxConverter

        # Get template path only if explicitly specified
        template_path = kwargs.get('template_path')
        if template_path and self.verbose:
            print(f"Using specified template: {template_path}")

        # Create formatter
        docx_formatter = DocxFormatter(template_path=template_path)

        # Create converter
        converter = MarkdownToDocxConverter(
            formatter=docx_formatter,
            verbose=self.verbose,
        )

        # Add table of contents if requested
        include_toc = kwargs.get('toc', False)

        # Convert the document
        converter.convert(
            markdown_path,
            output_path,
            template_path=template_path,
            include_toc=include_toc,
        )

        if self.verbose:
            print(f"Saved DOCX to {output_path}")
            if include_toc:
                print("Table of contents added. You may need to update it in Word.")

        return output_path

    def convert_pdf_to_markdown(
        self,
        pdf_path: str,
        output_path: str,
        **kwargs: Any,
    ) -> str:
        """
        Convert a PDF file to Markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.
        output_path : str
            Path to the output Markdown file.
        **kwargs : dict
            Additional arguments for the conversion.

        Returns
        -------
        str
            Path to the output Markdown file.
        """
        # Import here to avoid circular imports
        from docmark.converters.pdf import PdfToMarkdownConverter

        # Get images directory
        images_dir = kwargs.get('images_dir')
        if not images_dir:
            # Use 'media' folder in same directory as output file
            images_dir = os.path.join(os.path.dirname(output_path), "media")

        # Create converter
        converter = PdfToMarkdownConverter(
            llm_manager=self.llm_manager,
            image_processor=ImageProcessor(images_dir=images_dir),
            verbose=self.verbose,
        )

        # Convert the document
        markdown_content = converter.convert(pdf_path)

        # Format the markdown content
        markdown_content = self.markdown_formatter.format(
            markdown_content,
            use_llm=kwargs.get('use_llm', True),
        )

        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        if self.verbose:
            print(f"Saved Markdown to {output_path}")

        return output_path

    def convert_html_to_markdown(
        self,
        html_path: str,
        output_path: str,
        **kwargs: Any,
    ) -> str:
        """
        Convert an HTML file to Markdown.

        Parameters
        ----------
        html_path : str
            Path to the HTML file.
        output_path : str
            Path to the output Markdown file.
        **kwargs : dict
            Additional arguments for the conversion.

        Returns
        -------
        str
            Path to the output Markdown file.
        """
        # Import here to avoid circular imports
        from docmark.converters.html import HtmlToMarkdownConverter

        # Get images directory
        images_dir = kwargs.get('images_dir')
        if not images_dir:
            # Use 'media' folder in same directory as output file
            images_dir = os.path.join(os.path.dirname(output_path), "media")

        # Create converter
        converter = HtmlToMarkdownConverter(
            llm_manager=self.llm_manager,
            image_processor=ImageProcessor(images_dir=images_dir),
            verbose=self.verbose,
        )

        # Convert the document
        markdown_content = converter.convert(html_path)

        # Format the markdown content
        markdown_content = self.markdown_formatter.format(
            markdown_content,
            use_llm=kwargs.get('use_llm', True),
        )

        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        if self.verbose:
            print(f"Saved Markdown to {output_path}")

        return output_path


class DocMark:
    """Main DocMark class for document conversion."""

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: str = "openai",
        verbose: bool = False,
    ):
        """
        Initialize DocMark.

        Parameters
        ----------
        llm_api_key : str, optional
            API key for the LLM provider.
        llm_model : str, optional
            Model to use for LLM requests.
        llm_provider : str, optional
            LLM provider to use.
        verbose : bool, optional
            Whether to print verbose output.
        """
        self.processor = Processor(
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
            verbose=verbose,
        )
        self.verbose = verbose

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
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
        return self.processor.convert(input_path, output_path, **kwargs)

    def batch_convert(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.*",
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
        **kwargs : dict
            Additional arguments for the conversion.

        Returns
        -------
        List[str]
            List of paths to the output files.
        """
        # Find input files
        input_files = find_files(input_dir, pattern)

        if not input_files:
            if self.verbose:
                print(f"No files found matching pattern '{pattern}' in {input_dir}")
            return []

        # Set output directory
        if not output_dir:
            output_dir = input_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert each file
        output_files = []
        for input_file in input_files:
            # Determine output format
            output_format = kwargs.get('to_format')
            if not output_format:
                # Try to derive from input file
                input_format = get_file_type(input_file)
                if input_format == 'docx':
                    output_format = 'md'
                elif input_format == 'md':
                    output_format = 'docx'
                elif input_format == 'pdf':
                    output_format = 'md'
                elif input_format == 'html':
                    output_format = 'md'
                else:
                    # Skip unsupported formats
                    if self.verbose:
                        print(f"Skipping unsupported format: {input_file}")
                    continue

            # Determine output path
            filename = os.path.basename(input_file)
            name, _ = os.path.splitext(filename)
            output_file = os.path.join(output_dir, f"{name}.{output_format}")

            try:
                # Convert the file
                output_file = self.convert(
                    input_file,
                    output_file,
                    to_format=output_format,
                    **kwargs,
                )
                output_files.append(output_file)
            except (ValueError, FileNotFoundError, PermissionError, OSError) as e:
                if self.verbose:
                    print(f"Error converting {input_file}: {str(e)}")

        return output_files
