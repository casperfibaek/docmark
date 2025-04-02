"""
HTML to Markdown converter for DocMark.

This module provides functionality for converting HTML files to Markdown.
"""

import os
import re
import urllib.parse
import base64
from typing import Optional, Dict, Any
import requests

import markdownify
from bs4 import BeautifulSoup

from docmark.core.llm import LLMManager
from docmark.utils.image import ImageProcessor
from docmark.utils.text import fix_markdown_formatting, clean_html


class HtmlToMarkdownConverter:
    """Converter for HTML to Markdown."""

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        image_processor: Optional[ImageProcessor] = None,
        verbose: bool = False,
    ):
        """
        Initialize the HTML to Markdown converter.

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

    def convert(self, html_path: str) -> str:
        """
        Convert an HTML file to Markdown.

        Parameters
        ----------
        html_path : str
            Path to the HTML file.

        Returns
        -------
        str
            Markdown content.
        """
        if self.verbose:
            print(f"Converting HTML to Markdown: {html_path}")

        # Read the HTML file
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Clean the HTML
        html_content = clean_html(html_content)

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract and process images
        image_map = self._extract_images(soup, html_path)

        # Convert HTML to Markdown
        markdown_content = self._convert_html_to_markdown(soup)

        # Process images in markdown content
        markdown_content = self._process_images(markdown_content, image_map, html_path)

        # Fix markdown formatting
        markdown_content = fix_markdown_formatting(markdown_content)

        # Use LLM to enhance tables if available
        markdown_content = self._enhance_tables(markdown_content)

        return markdown_content

    def _extract_images(self, soup: BeautifulSoup, html_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract images from HTML.

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object containing the HTML content.
        html_path : str
            Path to the HTML file.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping original image URLs to processed image info.
        """
        image_map: Dict[str, Dict[str, Any]] = {}
        base_path = os.path.dirname(html_path)

        # Find all images
        for img in soup.find_all("img"):
            if "src" not in img.attrs:
                continue

            src = img["src"]
            alt = img.get("alt", "")

            try:
                # Process based on image source type
                if src.startswith("data:image/"):
                    self._process_data_uri_image(src, alt, html_path, image_map)
                elif src.startswith(("http://", "https://")):
                    self._process_remote_image(src, alt, html_path, image_map)
                else:
                    self._process_local_image(src, alt, base_path, html_path, image_map)
            except (IOError, ValueError, urllib.error.URLError) as e:
                if self.verbose:
                    print(f"Error processing image {src}: {str(e)}")

        return image_map

    def _process_data_uri_image(self, src: str, alt: str, html_path: str, image_map: Dict[str, Dict[str, Any]]) -> None:
        """Process a data URI image."""
        if self.verbose:
            print("Processing data URI image")

        # Extract image data from data URI
        data_uri_parts = src.split(",", 1)
        if len(data_uri_parts) != 2:
            return

        header, data = data_uri_parts
        content_type = header.split(";")[0].split(":")[1]

        try:
            image_data = base64.b64decode(data)

            # Save the image
            img_path = self.image_processor.extract_image(
                image_data, content_type, prefix="html_img", doc_filename=html_path
            )

            # Add to image map
            image_map[src] = {
                "path": img_path,
                "alt": alt,
            }

            if self.verbose:
                print(f"Extracted data URI image to {img_path}")
        except (base64.binascii.Error, ValueError, IOError) as e:
            if self.verbose:
                print(f"Error decoding data URI: {str(e)}")

    def _process_remote_image(self, src: str, alt: str, html_path: str, image_map: Dict[str, Dict[str, Any]]) -> None:
        """Process a remote image from a URL."""
        if self.verbose:
            print(f"Processing remote image: {src}")

        try:
            response = requests.get(src, stream=True, timeout=30)
            response.raise_for_status()

            # Get content type
            content_type = response.headers.get("content-type", "image/jpeg")

            # Save the image
            img_path = self.image_processor.extract_image(
                response.content, content_type, prefix="html_img", doc_filename=html_path
            )

            # Add to image map
            image_map[src] = {
                "path": img_path,
                "alt": alt,
            }

            if self.verbose:
                print(f"Downloaded remote image to {img_path}")
        except (requests.RequestException, IOError, ValueError) as e:
            if self.verbose:
                print(f"Error downloading remote image: {str(e)}")

    def _process_local_image(self, src: str, alt: str, base_path: str, html_path: str,
                             image_map: Dict[str, Dict[str, Any]]) -> None:
        """Process a local image file."""
        if self.verbose:
            print(f"Processing local image: {src}")

        # Resolve path
        img_path = src
        if not os.path.isabs(img_path):
            # Handle relative paths
            img_path = os.path.normpath(os.path.join(base_path, urllib.parse.unquote(img_path)))

        if not os.path.exists(img_path):
            if self.verbose:
                print(f"Local image not found: {img_path}")
            return

        # Read the image
        with open(img_path, "rb") as f:
            image_data = f.read()

        # Get content type
        content_type, _ = os.path.splitext(img_path)
        content_type = f"image/{content_type[1:]}" if content_type else "image/jpeg"

        # Save the image
        new_img_path = self.image_processor.extract_image(
            image_data, content_type, prefix="html_img", doc_filename=html_path
        )

        # Add to image map
        image_map[src] = {
            "path": new_img_path,
            "alt": alt,
        }

        if self.verbose:
            print(f"Copied local image to {new_img_path}")

    def _convert_html_to_markdown(self, soup: BeautifulSoup) -> str:
        """
        Convert HTML to Markdown.

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object containing the HTML content.

        Returns
        -------
        str
            Markdown content.
        """
        # Extract the body content
        body = soup.body or soup

        # Convert to string
        html_content = str(body)

        # Convert HTML to Markdown using markdownify
        markdown_content = markdownify.markdownify(html_content, heading_style="ATX")

        return markdown_content

    def _process_images(
        self, markdown_content: str, image_map: Dict[str, Dict[str, Any]], html_path: str
    ) -> str:
        """
        Process images in markdown content.

        Parameters
        ----------
        markdown_content : str
            Markdown content.
        image_map : Dict[str, Dict[str, Any]]
            Dictionary mapping original image URLs to processed image info.
        html_path : str
            Path to the source HTML file.

        Returns
        -------
        str
            Markdown content with processed images.
        """
        if not image_map:
            return markdown_content

        if self.verbose:
            print(f"Processing {len(image_map)} images in markdown content")

        # Process each image
        for src, img_info in image_map.items():
            img_path = img_info["path"]
            alt = img_info["alt"]

            # Get relative path for markdown
            rel_path = self.image_processor.get_relative_path(img_path, html_path)

            # Generate description if LLM manager is available and no alt text
            if self.llm_manager and not alt:
                try:
                    description = self.llm_manager.generate_image_description(img_path)
                    if description:
                        alt = description
                        if self.verbose:
                            print(f"Generated description for {img_path}: {description[:50]}...")
                except Exception as e:
                    if self.verbose:
                        print(f"Error generating description for {img_path}: {str(e)}")

            # Escape special characters in src for regex
            escaped_src = re.escape(src)

            # Replace image references in markdown
            img_pattern = r'!\[(.*?)\]\(' + escaped_src + r'\)'
            replacement = f'![{alt}]({rel_path})'

            markdown_content = re.sub(img_pattern, replacement, markdown_content)

        return markdown_content

    def _enhance_tables(self, markdown_content: str) -> str:
        """
        Enhance tables in markdown content using LLM.

        Parameters
        ----------
        markdown_content : str
            Markdown content.

        Returns
        -------
        str
            Markdown content with enhanced tables.
        """
        if not self.llm_manager:
            return markdown_content

        # Find all tables in the markdown content
        table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
        tables = re.findall(table_pattern, markdown_content)

        if not tables:
            return markdown_content

        if self.verbose:
            print(f"Enhancing {len(tables)} tables with LLM")

        # Process each table
        for table in tables:
            try:
                # Convert to HTML for better processing
                html_table = self._markdown_table_to_html(table)

                # Enhance with LLM
                enhanced_table = self.llm_manager.improve_table_conversion(html_table)

                # Replace in markdown content
                markdown_content = markdown_content.replace(table, enhanced_table)

                if self.verbose:
                    print("Enhanced table with LLM")
            except (ValueError, AttributeError, IndexError) as e:
                if self.verbose:
                    print(f"Error enhancing table: {str(e)}")

        return markdown_content

    def _markdown_table_to_html(self, markdown_table: str) -> str:
        """
        Convert a markdown table to HTML.

        Parameters
        ----------
        markdown_table : str
            Markdown table.

        Returns
        -------
        str
            HTML table.
        """
        lines = markdown_table.strip().split('\n')
        if len(lines) < 3:
            return markdown_table  # Not a valid table

        # Parse header
        header_cells = [cell.strip() for cell in lines[0].split('|')[1:-1]]

        # Parse separator line to determine alignment
        separator_cells = [cell.strip() for cell in lines[1].split('|')[1:-1]]
        alignments = []

        for cell in separator_cells:
            if cell.startswith(':') and cell.endswith(':'):
                alignments.append('center')
            elif cell.startswith(':'):
                alignments.append('left')
            elif cell.endswith(':'):
                alignments.append('right')
            else:
                alignments.append('left')  # Default alignment

        # Ensure alignments match header cells
        while len(alignments) < len(header_cells):
            alignments.append('left')

        # Build HTML table
        html = ['<table>']

        # Add header
        html.append('<thead>')
        html.append('<tr>')
        for i, cell in enumerate(header_cells):
            align = alignments[i] if i < len(alignments) else 'left'
            html.append(f'<th align="{align}">{cell}</th>')
        html.append('</tr>')
        html.append('</thead>')

        # Add body
        html.append('<tbody>')
        for i in range(2, len(lines)):
            row_cells = [cell.strip() for cell in lines[i].split('|')[1:-1]]
            html.append('<tr>')
            for j, cell in enumerate(row_cells):
                align = alignments[j] if j < len(alignments) else 'left'
                html.append(f'<td align="{align}">{cell}</td>')
            html.append('</tr>')
        html.append('</tbody>')

        html.append('</table>')

        return '\n'.join(html)
