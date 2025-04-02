"""
PDF to Markdown converter for DocMark.

This module provides functionality for converting PDF files to Markdown.
"""

import os
import re
import tempfile
from typing import Optional, Dict, Any, List, Tuple

from docmark.core.llm import LLMManager
from docmark.utils.image import ImageProcessor
from docmark.utils.text import fix_markdown_formatting


class PdfToMarkdownConverter:
    """Converter for PDF to Markdown."""

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        image_processor: Optional[ImageProcessor] = None,
        verbose: bool = False,
    ):
        """
        Initialize the PDF to Markdown converter.

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

    def convert(self, pdf_path: str) -> str:
        """
        Convert a PDF file to Markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        str
            Markdown content.
        """
        if self.verbose:
            print(f"Converting PDF to Markdown: {pdf_path}")

        # Extract text and images from PDF
        text_content, images = self._extract_pdf_content(pdf_path)

        # Use LLM to improve structure if available
        if self.llm_manager:
            markdown_content = self._structure_with_llm(text_content, pdf_path)
        else:
            # Basic conversion without LLM
            markdown_content = self._basic_conversion(text_content)

        # Process images
        markdown_content = self._process_images(markdown_content, images, pdf_path)

        # Fix markdown formatting
        markdown_content = fix_markdown_formatting(markdown_content)

        return markdown_content

    def _extract_pdf_content(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text and images from a PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        Tuple[str, List[Dict[str, Any]]]
            Extracted text content and list of images.
        """
        try:
            # Try to use pdfminer.six for extraction (primary method)
            from pdfminer.high_level import extract_text, extract_pages
            from pdfminer.layout import LAParams, LTImage, LTFigure
            from pdfminer.image import ImageWriter

            if self.verbose:
                print("Using pdfminer.six for PDF extraction")

            # Configure layout parameters for text extraction
            laparams = LAParams(
                line_margin=0.5,
                char_margin=2.0,
                all_texts=True
            )

            # Extract text
            text_content = extract_text(
                pdf_path,
                laparams=laparams,
                codec="utf-8"
            )

            # Extract images
            images = self._extract_images_with_pdfminer(pdf_path)

            if not text_content.strip():
                if self.verbose:
                    print("No text extracted with pdfminer.six, trying alternative method")
                return self._extract_with_pypdf(pdf_path)

            return text_content, images

        except ImportError:
            if self.verbose:
                print("pdfminer.six not available, trying alternative extraction method")
            return self._extract_with_pypdf(pdf_path)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting with pdfminer.six: {str(e)}")
            return self._extract_with_pypdf(pdf_path)

    def _extract_images_with_pdfminer(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file using pdfminer.six.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        List[Dict[str, Any]]
            List of extracted images.
        """
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTImage, LTFigure
        from pdfminer.image import ImageWriter

        extracted_images = []

        try:
            # Create a temporary directory for image extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Setup image writer
                image_writer = ImageWriter(temp_dir)

                # Extract pages and process images
                for page_num, page_layout in enumerate(extract_pages(pdf_path)):
                    # Process all figures and images on the page
                    for element in page_layout:
                        if isinstance(element, (LTImage, LTFigure)):
                            # Try to extract images from figures
                            img_path = None
                            try:
                                # For direct images (LTImage)
                                if isinstance(element, LTImage):
                                    img_path = image_writer.export_image(element)
                                # For figures (LTFigure) which might contain images
                                elif isinstance(element, LTFigure):
                                    for item in element:
                                        if isinstance(item, LTImage):
                                            img_path = image_writer.export_image(item)
                                            break
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error extracting image: {str(e)}")
                                continue

                            # Process the extracted image if available
                            if img_path:
                                try:
                                    # Full path to the temporary image
                                    full_img_path = os.path.join(temp_dir, img_path)

                                    # Read image data
                                    with open(full_img_path, "rb") as img_file:
                                        image_data = img_file.read()

                                    # Process and save the image
                                    final_img_path = self.image_processor.extract_image(
                                        image_data,
                                        "image/png",  # Assume PNG for simplicity
                                        prefix=f"pdf_img_{page_num}",
                                        doc_filename=pdf_path
                                    )

                                    # Add to images list
                                    extracted_images.append({
                                        "path": final_img_path,
                                        "page": page_num,
                                        "bbox": [element.x0, element.y0, element.x1, element.y1],
                                    })

                                except Exception as e:
                                    if self.verbose:
                                        print(f"Error processing image {img_path}: {str(e)}")

            return extracted_images

        except Exception as e:
            if self.verbose:
                print(f"Error in image extraction: {str(e)}")
            return []

    def _extract_with_pypdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text and images using pypdf.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        Tuple[str, List[Dict[str, Any]]]
            Extracted text content and list of images.
        """
        try:
            # Try to use pypdf for extraction
            from pypdf import PdfReader

            if self.verbose:
                print("Using pypdf for PDF extraction")

            reader = PdfReader(pdf_path)
            text_content = ""

            # Extract text from each page
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n\n"

            # For now, we don't extract images with pypdf
            # This would require more complex handling
            images = []

            if not text_content.strip():
                # If no text was extracted, try alternative method
                return self._extract_with_alternative(pdf_path)

            return text_content, images

        except ImportError:
            if self.verbose:
                print("pypdf not available, trying alternative extraction method")
            return self._extract_with_alternative(pdf_path)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting with pypdf: {str(e)}")
            return self._extract_with_alternative(pdf_path)

    def _extract_with_alternative(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text and images using an alternative method.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        Tuple[str, List[Dict[str, Any]]]
            Extracted text content and list of images.
        """
        try:
            # Try to use pdfplumber
            import pdfplumber

            if self.verbose:
                print("Using pdfplumber for PDF extraction")

            with pdfplumber.open(pdf_path) as pdf:
                text_content = ""
                images = []

                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"

                    # Extract images
                    for img in page.images:
                        # Save image to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(img["stream"].get_data())
                            img_path = tmp.name

                        # Process and save the image
                        try:
                            with open(img_path, "rb") as img_file:
                                image_data = img_file.read()

                            final_img_path = self.image_processor.extract_image(
                                image_data,
                                "image/png",
                                prefix=f"pdf_img_{page_num}",
                                doc_filename=pdf_path
                            )

                            # Add to images list
                            images.append({
                                "path": final_img_path,
                                "page": page_num,
                                "bbox": img["bbox"],
                            })

                            # Clean up temp file
                            os.unlink(img_path)

                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing image: {str(e)}")

                return text_content, images

        except ImportError:
            if self.verbose:
                print("pdfplumber not available, trying OCR extraction")
            return self._extract_with_ocr(pdf_path)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting with pdfplumber: {str(e)}")
            return self._extract_with_ocr(pdf_path)

    def _extract_with_ocr(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text using OCR as a last resort.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        Tuple[str, List[Dict[str, Any]]]
            Extracted text content and list of images.
        """
        try:
            # Try to use pdf2image and pytesseract
            from pdf2image import convert_from_path
            import pytesseract

            if self.verbose:
                print("Using OCR (pytesseract) for PDF extraction")

            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text_content = ""
            extracted_images = []

            for i, img in enumerate(images):
                # Extract text with OCR
                page_text = pytesseract.image_to_string(img)
                text_content += page_text + "\n\n"

                # Save the page image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    img.save(tmp.name, "PNG")
                    img_path = tmp.name

                # Process and save the image
                try:
                    with open(img_path, "rb") as img_file:
                        image_data = img_file.read()

                    final_img_path = self.image_processor.extract_image(
                        image_data,
                        "image/png",
                        prefix=f"pdf_page_{i}",
                        doc_filename=pdf_path
                    )

                    # Add to images list
                    extracted_images.append({
                        "path": final_img_path,
                        "page": i,
                        "is_page": True,
                    })

                    # Clean up temp file
                    os.unlink(img_path)

                except Exception as e:
                    if self.verbose:
                        print(f"Error processing page image: {str(e)}")

            return text_content, extracted_images

        except ImportError:
            if self.verbose:
                print("OCR libraries not available, returning empty content")
            return "", []
        except Exception as e:
            if self.verbose:
                print(f"Error extracting with OCR: {str(e)}")
            return "", []

    def _structure_with_llm(self, text_content: str, pdf_path: str) -> str:
        """
        Use LLM to structure the extracted text.

        Parameters
        ----------
        text_content : str
            Extracted text content.
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        str
            Structured Markdown content.
        """
        if self.verbose:
            print("Using LLM to structure PDF content")

        try:
            # Extract document structure using LLM
            structure = self.llm_manager.extract_structure_from_pdf(text_content)

            # Build markdown content based on structure
            markdown_lines = []

            # Add title if available
            if structure.get("title"):
                markdown_lines.append(f"# {structure['title']}\n")

            # Process the text content with the extracted structure
            remaining_text = text_content

            # Process headings
            for heading in structure.get("headings", []):
                heading_text = heading.get("text", "")
                level = heading.get("level", 1)

                if heading_text:
                    # Find the heading in the text
                    heading_pos = remaining_text.find(heading_text)
                    if heading_pos >= 0:
                        # Add text before the heading
                        before_text = remaining_text[:heading_pos].strip()
                        if before_text:
                            markdown_lines.append(before_text + "\n\n")

                        # Add the heading
                        markdown_lines.append(f"{'#' * level} {heading_text}\n\n")

                        # Update remaining text
                        remaining_text = remaining_text[heading_pos + len(heading_text):].strip()

            # Add any remaining text
            if remaining_text:
                markdown_lines.append(remaining_text)

            # Join all lines
            markdown_content = "\n".join(markdown_lines)

            # Use LLM to enhance the markdown
            markdown_content = self.llm_manager.enhance_markdown(markdown_content)

            return markdown_content

        except Exception as e:
            if self.verbose:
                print(f"Error structuring with LLM: {str(e)}")
            # Fall back to basic conversion
            return self._basic_conversion(text_content)

    def _basic_conversion(self, text_content: str) -> str:
        """
        Perform basic conversion of text to Markdown.

        Parameters
        ----------
        text_content : str
            Extracted text content.

        Returns
        -------
        str
            Basic Markdown content.
        """
        if self.verbose:
            print("Performing basic PDF to Markdown conversion")

        # Split text into lines
        lines = text_content.split("\n")
        markdown_lines = []

        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append("")
                continue

            # Check if line looks like a heading (all caps, short line)
            if line.isupper() and len(line) < 100:
                markdown_lines.append(f"## {line}")
            # Check if line looks like a subheading (starts with number or has colon)
            elif re.match(r'^\d+\.', line) or ': ' in line[:20]:
                markdown_lines.append(f"### {line}")
            # Regular text
            else:
                markdown_lines.append(line)

        # Join lines
        markdown_content = "\n".join(markdown_lines)

        # Fix common PDF extraction issues
        markdown_content = re.sub(r'(\w)-\n(\w)', r'\1\2', markdown_content)  # Fix hyphenation
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excessive newlines

        return markdown_content

    def _process_images(
        self, markdown_content: str, images: List[Dict[str, Any]], pdf_path: str
    ) -> str:
        """
        Process images and add them to the markdown content.

        Parameters
        ----------
        markdown_content : str
            Markdown content.
        images : List[Dict[str, Any]]
            List of extracted images.
        pdf_path : str
            Path to the source PDF file.

        Returns
        -------
        str
            Markdown content with images.
        """
        if not images:
            return markdown_content

        if self.verbose:
            print(f"Processing {len(images)} images from PDF")

        # Add images to the markdown content
        for _i, img in enumerate(images):
            img_path = img["path"]
            page = img.get("page", 0)

            # Get relative path for markdown
            rel_path = self.image_processor.get_relative_path(img_path, pdf_path)

            # Generate description if LLM manager is available
            alt_text = f"Image from page {page+1}"
            if self.llm_manager:
                try:
                    description = self.llm_manager.generate_image_description(img_path)
                    if description:
                        alt_text = description
                except Exception as e:
                    if self.verbose:
                        print(f"Error generating description for {img_path}: {str(e)}")

            # Create markdown image tag
            img_markdown = f"\n\n![{alt_text}]({rel_path})\n\n"

            # If it's a full page image, add it at the end
            if img.get("is_page", False):
                markdown_content += img_markdown
            else:
                # Try to insert image at a reasonable position based on page number
                # This is a heuristic and might not be perfect
                lines = markdown_content.split("\n")
                total_lines = len(lines)
                target_position = min(int(total_lines * (page / max(len(images), 1))), total_lines - 1)

                # Find the nearest paragraph break
                while target_position < total_lines - 1 and lines[target_position].strip():
                    target_position += 1

                # Insert the image
                lines.insert(target_position, img_markdown)
                markdown_content = "\n".join(lines)

        return markdown_content
