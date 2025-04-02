"""
DOCX to Markdown converter for DocMark.

This module provides functionality for converting DOCX files to Markdown.
"""

import re
from typing import Optional, Dict, Any, Tuple

import mammoth
import markdownify

from docmark.core.llm import LLMManager
from docmark.utils.image import ImageProcessor
from docmark.utils.text import fix_markdown_formatting


class DocxToMarkdownConverter:
    """Converter for DOCX to Markdown."""

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        image_processor: Optional[ImageProcessor] = None,
        verbose: bool = False,
    ):
        """
        Initialize the DOCX to Markdown converter.

        Parameters
        ----------
        llm_manager : LLMManager, optional
            LLM manager for enhanced conversion.
        image_processor : ImageProcessor, optional
            Image processor for handling images.
        verbose : bool, optional
            Whether to print verbose output.
        """
        self.llm_manager = llm_manager
        self.image_processor = image_processor or ImageProcessor()
        self.verbose = verbose

    def convert(self, docx_path: str) -> str:
        """
        Convert a DOCX file to Markdown.

        Parameters
        ----------
        docx_path : str
            Path to the DOCX file.

        Returns
        -------
        str
            Markdown content.
        """
        if self.verbose:
            print(f"Converting DOCX to Markdown: {docx_path}")

        # Extract HTML content from DOCX
        html_content, image_buffer = self._extract_html_with_images(docx_path)

        # Convert HTML to Markdown
        markdown_content = self._convert_html_to_markdown(html_content)

        # Process images
        markdown_content = self._process_images(
            markdown_content, image_buffer, docx_path
        )

        # Fix markdown formatting
        markdown_content = fix_markdown_formatting(markdown_content)

        return markdown_content

    def _extract_html_with_images(self, docx_path: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Extract HTML content and images from a DOCX file.

        Parameters
        ----------
        docx_path : str
            Path to the DOCX file.

        Returns
        -------
        Tuple[str, Dict[str, Dict[str, Any]]]
            HTML content and image buffer.
        """
        image_buffer = {}

        # Define image converter function
        def image_converter(image):
            return self._convert_mammoth_image(image, docx_path, image_buffer)

        # Convert DOCX to HTML with image extraction
        with open(docx_path, "rb") as docx_file:
            result = mammoth.convert_to_html(
                docx_file, convert_image=mammoth.images.img_element(image_converter)
            )
            html_content = result.value

            if self.verbose and result.messages:
                for message in result.messages:
                    print(f"Mammoth message: {message}")

        return html_content, image_buffer

    def _convert_mammoth_image(
        self, image: Any, docx_path: str, image_buffer: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Convert an image extracted by mammoth.

        Parameters
        ----------
        image : Any
            Image object from mammoth.
        docx_path : str
            Path to the source DOCX file.
        image_buffer : Dict[str, Dict[str, Any]]
            Buffer to store information about the extracted image.

        Returns
        -------
        Dict[str, str]
            Dictionary with the image source path for HTML.
        """
        try:
            with image.open() as image_bytes:
                content_type = image.content_type
                image_data = image_bytes.read()

                # Extract and save the image with document filename for naming
                image_path = self.image_processor.extract_image(
                    image_data, content_type, prefix="docx_img", doc_filename=docx_path
                )

                # Store reference to extracted image
                image_buffer[image_path] = {
                    "content_type": content_type,
                    "data": None,  # We don't need to keep the data in memory
                }

                # Get relative path for markdown
                rel_path = self.image_processor.get_relative_path(image_path, docx_path)

                if self.verbose:
                    print(f"Extracted image: {rel_path}")

                return {"src": rel_path}
        except Exception as e:
            if self.verbose:
                print(f"Error processing image: {str(e)}")
            return {"src": "image_extraction_failed.png"}

    def _convert_html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to Markdown.

        Parameters
        ----------
        html_content : str
            HTML content.

        Returns
        -------
        str
            Markdown content.
        """
        # Convert HTML to Markdown using markdownify
        markdown_content = markdownify.markdownify(html_content, heading_style="ATX")

        return markdown_content

    def _process_images(
        self, markdown_content: str, image_buffer: Dict[str, Dict[str, Any]], docx_path: str
    ) -> str:
        """
        Process images in markdown content.

        Parameters
        ----------
        markdown_content : str
            Markdown content.
        image_buffer : Dict[str, Dict[str, Any]]
            Buffer with extracted images.
        docx_path : str
            Path to the source DOCX file.

        Returns
        -------
        str
            Markdown content with processed images.
        """
        # Generate image descriptions if LLM manager is available
        if self.llm_manager and image_buffer:
            if self.verbose:
                print(f"Generating descriptions for {len(image_buffer)} images")

            for img_path in image_buffer:
                try:
                    # Find markdown references to this image
                    rel_path = self.image_processor.get_relative_path(img_path, docx_path)

                    # Create pattern to match image in markdown
                    pattern = r"!\[(.*?)\]\(" + re.escape(rel_path) + r"\)"

                    # Also check for path without ./ prefix
                    alt_rel_path = rel_path
                    if alt_rel_path.startswith("./"):
                        alt_rel_path = alt_rel_path[2:]
                        alt_pattern = r"!\[(.*?)\]\(" + re.escape(alt_rel_path) + r"\)"
                        pattern = f"({pattern}|{alt_pattern})"

                    if re.search(pattern, markdown_content):
                        # Generate description
                        description = self.llm_manager.generate_image_description(img_path)

                        if self.verbose:
                            print(f"Generated description for {rel_path}: {description[:50]}...")

                        # Add description to the image alt text
                        replacement = f"![{description}]({rel_path})"

                        # Handle both path formats (with and without ./)
                        if alt_rel_path != rel_path:
                            markdown_content = re.sub(
                                r"!\[(.*?)\]\(" + re.escape(alt_rel_path) + r"\)",
                                replacement,
                                markdown_content,
                            )

                        markdown_content = re.sub(
                            r"!\[(.*?)\]\(" + re.escape(rel_path) + r"\)",
                            replacement,
                            markdown_content,
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"Error generating description for {img_path}: {str(e)}")

        return markdown_content
