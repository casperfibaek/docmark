"""
LLM integration module for DocMark.

This module provides functionality for interacting with language models
for various tasks such as formatting enhancement, image description, and
content recognition.
"""

import os
import base64
import mimetypes
import json
from typing import Optional, Dict, Any


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM provider.

        Parameters
        ----------
        api_key : str, optional
            API key for the LLM provider. If not provided, will try to get from environment.
        model : str, optional
            Model to use for LLM requests. If not provided, will use a default model.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or "o3-mini"  # Default to o3-mini as specified
        self.client: Any = None

    def initialize(self) -> None:
        """Initialize the LLM client."""
        raise NotImplementedError("Subclasses must implement initialize()")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM.
        **kwargs : dict
            Additional arguments to pass to the LLM.

        Returns
        -------
        str
            The generated text.
        """
        raise NotImplementedError("Subclasses must implement generate_text()")

    def generate_image_description(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a description for an image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str, optional
            Custom prompt for the LLM.

        Returns
        -------
        str
            The generated description.
        """
        raise NotImplementedError("Subclasses must implement generate_image_description()")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            # Import inside the method to avoid errors when the package is not installed
            # but allow for type hinting at the class level
            import openai
  # type: ignore
            self.client = openai.Client(api_key=self.api_key)
        except ImportError as exc:
            raise ImportError("OpenAI package is required. Install with 'pip install openai'") from exc

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI.

        Parameters
        ----------
        prompt : str
            The prompt to send to OpenAI.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        str
            The generated text.
        """
        if not self.client:
            self.initialize()

        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4000)  # Increased from 1000 to reduce chunks needed

        # Prepare the API request parameters
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
        }

        # Handle o3-mini model parameter differences
        if "o3-mini" in self.model:
            # o3-mini uses max_completion_tokens instead of max_tokens
            request_params["max_completion_tokens"] = max_tokens
            # o3-mini does not support temperature parameter
        else:
            request_params["max_tokens"] = max_tokens
            request_params["temperature"] = temperature

        response = self.client.chat.completions.create(**request_params)

        return response.choices[0].message.content

    def generate_image_description(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a description for an image using OpenAI.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str, optional
            Custom prompt for the LLM.

        Returns
        -------
        str
            The generated description.
        """
        if not self.client:
            self.initialize()

        # Default prompt for image description
        if prompt is None:
            prompt = "Describe this image in detail. Focus on the main elements and context."

        # Read image and convert to base64
        with open(image_path, "rb") as image_file:
            content_type, _ = mimetypes.guess_type(image_path)
            content_type = content_type or "image/jpeg"
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_image}"}}
                    ]
                }
            ]
        }

        # Handle o3-mini model parameter differences
        if "o3-mini" in self.model:
            # o3-mini uses max_completion_tokens instead of max_tokens
            request_params["max_completion_tokens"] = 300
            # o3-mini does not support temperature parameter
        else:
            request_params["max_tokens"] = 300
            request_params["temperature"] = 0.7  # Default temperature for image descriptions

        # Generate description
        response = self.client.chat.completions.create(**request_params)

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""

    def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        try:
            # Import inside the method to avoid errors when the package is not installed
            # but allow for type hinting at the class level
            import anthropic
  # type: ignore
            self.client = anthropic.Client(api_key=self.api_key)
        except ImportError as exc:
            raise ImportError("Anthropic package is required. Install with 'pip install anthropic'") from exc

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic.

        Parameters
        ----------
        prompt : str
            The prompt to send to Anthropic.
        **kwargs : dict
            Additional arguments to pass to the Anthropic API.

        Returns
        -------
        str
            The generated text.
        """
        if not self.client:
            self.initialize()

        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)

        response = self.client.messages.create(
            model=self.model,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.content[0].text

    def generate_image_description(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a description for an image using Anthropic.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str, optional
            Custom prompt for the LLM.

        Returns
        -------
        str
            The generated description.
        """
        if not self.client:
            self.initialize()

        # Default prompt for image description
        if prompt is None:
            prompt = "Describe this image in detail. Focus on the main elements and context."

        # Read image and convert to base64
        with open(image_path, "rb") as image_file:
            content_type, _ = mimetypes.guess_type(image_path)
            content_type = content_type or "image/jpeg"
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        # Generate description
        response = self.client.messages.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": content_type, "data": base64_image}}
                    ]
                }
            ],
            max_tokens=300,
        )

        return response.content[0].text


class LLMManager:
    """Manager for LLM providers."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the LLM manager.

        Parameters
        ----------
        provider : str
            The LLM provider to use. Options: "openai", "anthropic".
        api_key : str, optional
            API key for the LLM provider. If not provided, will try to get from environment.
        model : str, optional
            Model to use for LLM requests. If not provided, will use a default model.
        """
        self.provider_name = provider.lower()
        self.api_key = api_key
        self.model = model
        self.provider = self._get_provider()

    def _get_provider(self) -> LLMProvider:
        """
        Get the LLM provider.

        Returns
        -------
        LLMProvider
            The LLM provider.
        """
        if self.provider_name == "openai":
            return OpenAIProvider(api_key=self.api_key, model=self.model)
        if self.provider_name == "anthropic":
            return AnthropicProvider(api_key=self.api_key, model=self.model)

        raise ValueError(f"Unsupported provider: {self.provider_name}")

    def initialize(self) -> None:
        """Initialize the LLM provider."""
        self.provider.initialize()

    def enhance_markdown(self, markdown_content: str) -> str:
        """
        Enhance Markdown content using LLM.

        Parameters
        ----------
        markdown_content : str
            The Markdown content to enhance.

        Returns
        -------
        str
            The enhanced Markdown content.
        """
        # For large documents, process in chunks to ensure complete formatting
        if len(markdown_content) > 10000:  # If content is larger than ~10k chars
            return self._enhance_markdown_in_chunks(markdown_content)

        prompt = f"""
        You are a markdown formatting expert. Please fix and enhance the following markdown text:
        - Ensure proper GitHub Flavored Markdown formatting
        - Fix headings (ensure proper # formatting)
        - Fix list formatting (ensure proper indentation and bullet points)
        - Fix code blocks (ensure proper ``` with language specification)
        - Fix table formatting (ensure proper alignment)
        - Fix image and link syntax
        - Preserve all content and meanings
        - Do not add or remove information
        - Return the COMPLETE markdown text with your formatting improvements
        - IMPORTANT: DO NOT wrap your response in triple backticks (```) or add markdown code block syntax
        - IMPORTANT: Return ONLY the formatted content, not enclosed in ```markdown or ``` markers

        Here is the markdown text to enhance:

        {markdown_content}
        """

        system_message = """
        You are a helpful assistant specialized in fixing markdown formatting.
        You fix issues in markdown text while preserving the content and meaning.
        You do not add or remove information, only fix formatting issues.
        You follow GitHub Flavored Markdown specifications.
        You MUST return the entire document, keeping ALL content intact.
        IMPORTANT: DO NOT wrap your response in triple backticks (```) or markdown code block syntax.
        Your output will be directly used as markdown content, so return only the formatted content itself.
        """

        # Create parameters with correct token limit format for the model
        text_gen_params = {
            "prompt": prompt,
            "system_message": system_message,
            "max_tokens": 4000,  # Increase token limit for larger documents
        }

        # Only add temperature parameter for models that support it
        if not "o3-mini" in getattr(self.provider, 'model', ''):
            text_gen_params["temperature"] = 0.3  # Lower temperature for more consistent results

        response = self.provider.generate_text(**text_gen_params)

        # Strip any markdown code block markers that might have been added despite instructions
        return self._strip_markdown_code_markers(response)

    def _enhance_markdown_in_chunks(self, markdown_content: str) -> str:
        """
        Process large markdown content in chunks to ensure complete formatting.

        Parameters
        ----------
        markdown_content : str
            The large markdown content to enhance.

        Returns
        -------
        str
            The enhanced markdown content with all chunks combined.
        """
        print(f"Processing large document in chunks (length: {len(markdown_content)})")

        # When using o3-mini model, use the more conservative approach directly
        # since it has shown to handle large content better with our overlap method
        if "o3-mini" in getattr(self.provider, 'model', ''):
            print("Using conservative chunking strategy for o3-mini model")
            return self._ensure_complete_markdown_formatting(markdown_content)

        # For other models, try the standard approach first
        # Save the original content length for verification
        original_content_length = len(markdown_content.strip())

        # Split the markdown content by sections (looking for headers)
        import re

        # First try to split by main headers (# or ## level headers)
        chunks = re.split(r'(^|\n)(#|##)\s+', markdown_content)

        if len(chunks) <= 2:  # Not enough headers to split effectively
            # Fall back to splitting by paragraphs
            chunks = re.split(r'\n\n+', markdown_content)

        # Keep track of all chunks for verification
        all_original_chunks = []
        processed_chunks = []
        current_chunk = ""

        # Use smaller chunks for better reliability
        max_chunk_size = 3000  # Reduced from 4000

        for chunk in chunks:
            if not chunk.strip():  # Skip empty chunks
                continue

            all_original_chunks.append(chunk)  # Save original for verification

            if len(current_chunk) + len(chunk) < max_chunk_size:
                current_chunk += chunk
            else:
                if current_chunk:
                    # Process the chunk and track results
                    processed_result = self._process_single_chunk(current_chunk)
                    processed_chunks.append(processed_result)
                current_chunk = chunk

        # Process the last chunk
        if current_chunk:
            processed_result = self._process_single_chunk(current_chunk)
            processed_chunks.append(processed_result)

        # Combine all processed chunks
        combined_result = "\n\n".join(processed_chunks)

        # Verify that the combined result isn't significantly shorter than the original
        result_length = len(combined_result.strip())
        length_ratio = result_length / original_content_length if original_content_length > 0 else 0

        # If the formatted content is significantly shorter (less than 90% of original),
        # there might be missing chunks - verify each chunk was processed
        if length_ratio < 0.9:
            print(f"Warning: Formatted content length ({result_length}) is significantly shorter than original ({original_content_length}).")
            print(f"Performing additional verification and fallback processing...")

            # If we lost significant content, use a simpler but more robust chunking approach
            return self._ensure_complete_markdown_formatting(markdown_content)

        return combined_result

    def _process_single_chunk(self, chunk: str) -> str:
        """
        Process a single chunk of markdown content.

        Parameters
        ----------
        chunk : str
            A chunk of markdown content.

        Returns
        -------
        str
            The enhanced chunk.
        """
        # Save original chunk length for verification
        original_chunk_length = len(chunk.strip())

        prompt = f"""
        You are a markdown formatting expert. Please fix and enhance this section of markdown text:
        - Ensure proper GitHub Flavored Markdown formatting
        - Fix headings (ensure proper # formatting)
        - Fix list formatting (ensure proper indentation and bullet points)
        - Fix code blocks (ensure proper ``` with language specification)
        - Fix table formatting (ensure proper alignment)
        - Fix image and link syntax
        - Preserve all content and meanings
        - Do not add or remove information
        - Keep ALL text content intact
        - IMPORTANT: DO NOT wrap your response in triple backticks (```) or add markdown code block syntax
        - IMPORTANT: Return ONLY the formatted content, not enclosed in ```markdown or ``` markers
        - CRITICALLY IMPORTANT: You MUST include the ENTIRE text provided, without omitting any content

        Here is the markdown section to enhance:

        {chunk}
        """

        system_message = """
        You are a helpful assistant specialized in fixing markdown formatting.
        You fix formatting issues while preserving ALL content and meaning.
        You MUST return the COMPLETE text that was provided to you, with improved formatting only.
        Do not summarize, condense, or remove any content.
        You must include every single paragraph, heading, list item, and sentence.
        IMPORTANT: DO NOT wrap your response in triple backticks (```) or markdown code block syntax.
        Your output will be directly used as markdown content, so return only the formatted content itself.
        """

        try:
            # Create parameters object appropriate for the model being used
            text_gen_params = {
                "prompt": prompt,
                "system_message": system_message,
                "max_tokens": 2000,
            }

            # Only add temperature parameter for models that support it
            if not "o3-mini" in getattr(self.provider, 'model', ''):
                text_gen_params["temperature"] = 0.3

            response = self.provider.generate_text(**text_gen_params)

            # Strip any markdown code block markers that might have been added despite instructions
            response = self._strip_markdown_code_markers(response)

            # Verify the processed chunk is not significantly shorter than the original
            response_length = len(response.strip())

            # If the processed chunk is significantly shorter (less than 85% of original),
            # fall back to the original chunk to ensure no content is lost
            if response_length < original_chunk_length * 0.85:
                print(f"Warning: Processed chunk length ({response_length}) is much shorter than original ({original_chunk_length}). Using original to preserve content.")
                return chunk

            return response
        except Exception as e:
            # If processing fails, return the original chunk
            print(f"Warning: Failed to process markdown chunk: {str(e)}")
            # Log the model being used for diagnosis
            print(f"Model being used: {getattr(self.provider, 'model', 'unknown')}")
            return chunk

    def _strip_markdown_code_markers(self, text: str) -> str:
        """
        Strip markdown code block markers from text.

        Parameters
        ----------
        text : str
            Text that might contain markdown code block markers.

        Returns
        -------
        str
            Text with markdown code block markers removed.
        """
        import re

        # Check if the text starts with ```markdown or similar and ends with ```
        text = re.sub(r'^\s*```(?:markdown)?\s*\n', '', text)
        text = re.sub(r'\n\s*```\s*$', '', text)

        return text

    def _ensure_complete_markdown_formatting(self, markdown_content: str) -> str:
        """
        Fail-safe method to process markdown content in a way that guarantees
        no content is lost, using a more conservative approach with overlapping chunks.

        Parameters
        ----------
        markdown_content : str
            The markdown content to format.

        Returns
        -------
        str
            The formatted markdown content with guaranteed completeness.
        """
        import re
        from docmark.utils import text as text_utils

        print(f"Using conservative formatting approach with overlapping chunks for content of length {len(markdown_content)}")

        # For o3-mini model, we'll use even smaller chunks with greater overlap
        # to ensure each piece is processed reliably
        is_o3_mini = "o3-mini" in getattr(self.provider, 'model', '')

        # Adjust chunk size based on model
        if is_o3_mini:
            chunk_size = 1500  # Smaller chunks for o3-mini
            overlap = 400      # Greater overlap
            print(f"Using smaller chunks (size: {chunk_size}, overlap: {overlap}) for o3-mini model")
        else:
            chunk_size = 3000  # Standard chunk size for other models
            overlap = 500      # Standard overlap

        # Store original content length for verification
        original_length = len(markdown_content)

        # If content is small enough, try processing it in one go
        if original_length <= chunk_size:
            try:
                formatted = self._process_single_chunk(markdown_content)

                # Verify the result isn't significantly shorter
                if len(formatted) < original_length * 0.9:
                    print(f"Warning: Formatted output is too short. Using original content with basic formatting.")
                    return text_utils.fix_markdown_formatting(markdown_content)

                return formatted
            except Exception as e:
                print(f"Error formatting small content: {str(e)}")
                return text_utils.fix_markdown_formatting(markdown_content)

        print(f"Splitting content into {chunk_size} byte chunks with {overlap} byte overlap")

        # For larger content, use overlapping chunks
        chunks = []
        formatted_chunks = []

        # Create overlapping chunks
        for i in range(0, original_length, chunk_size - overlap):
            # Get chunk with overlap
            end = min(i + chunk_size, original_length)
            chunk = markdown_content[i:end]
            chunks.append(chunk)

        print(f"Created {len(chunks)} chunks for processing")

        # Process each chunk
        successful_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                print(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)})")
                formatted_chunk = self._process_single_chunk(chunk)

                # Verify chunk was successfully processed
                if len(formatted_chunk.strip()) < len(chunk.strip()) * 0.8:
                    print(f"Warning: Chunk {i+1} formatting produced much shorter result - using original")
                    formatted_chunk = chunk
                else:
                    successful_chunks += 1

                # Special handling to stitch chunks together
                if i > 0 and len(formatted_chunk) > 0:
                    # Find where the overlap begins in the formatted chunk
                    # To avoid duplicating content, we'll identify the overlap section
                    prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]

                    # Look for a significant portion of the overlap text in the formatted chunk
                    # This is a simplification - in a production system, you might use more
                    # sophisticated text alignment algorithms
                    overlap_marker = prev_end[:100]  # Use the first part of the overlap as a marker

                    if overlap_marker and len(overlap_marker.strip()) > 20:
                        # Try to find where this marker appears in the formatted chunk
                        try:
                            # Use fuzzy matching to find approximate position of overlap
                            from difflib import SequenceMatcher

                            # Look through the first part of the formatted chunk for the marker
                            best_match_ratio = 0
                            best_match_pos = 0

                            # Check chunks of the formatted text to find best match for overlap
                            search_area = formatted_chunk[:min(500, len(formatted_chunk))]
                            step = 10

                            for j in range(0, len(search_area) - 20, step):
                                if j + 100 > len(search_area):
                                    break

                                test_segment = search_area[j:j+100]
                                ratio = SequenceMatcher(None, overlap_marker, test_segment).ratio()

                                if ratio > best_match_ratio:
                                    best_match_ratio = ratio
                                    best_match_pos = j

                            # If we found a good match, skip that portion
                            if best_match_ratio > 0.6:  # Threshold for considering it a match
                                print(f"Found overlap with ratio {best_match_ratio:.2f} at position {best_match_pos}")
                                formatted_chunk = formatted_chunk[best_match_pos:]
                            else:
                                print(f"No significant overlap found (best ratio: {best_match_ratio:.2f})")
                                # Use a small buffer to avoid duplication when no good match is found
                                formatted_chunk = formatted_chunk[min(50, len(formatted_chunk)//10):]

                        except Exception as e:
                            print(f"Warning: Error in overlap detection: {str(e)}")
                            # If error in overlap detection, keep a small buffer to avoid duplication
                            formatted_chunk = formatted_chunk[min(50, len(formatted_chunk)//10):]

                formatted_chunks.append(formatted_chunk)
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                # Use original chunk on error
                formatted_chunks.append(chunk)

        # Combine all chunks
        result = "".join(formatted_chunks)
        result_length = len(result.strip())

        # Report on processing stats
        print(f"Chunk processing completed: {successful_chunks}/{len(chunks)} chunks successfully processed")
        print(f"Original length: {original_length}, Result length: {result_length}, Ratio: {result_length/original_length:.2f}")

        # Final verification
        if len(result) < original_length * 0.9:
            print(f"WARNING: Final formatted content length ({len(result)}) is significantly shorter than original ({original_length}).")
            print("Falling back to original content with basic formatting only.")

            # Apply only basic formatting without LLM
            result = text_utils.fix_markdown_formatting(markdown_content)

        return result

    def generate_image_description(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a description for an image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str, optional
            Custom prompt for the LLM.

        Returns
        -------
        str
            The generated description.
        """
        if prompt is None:
            prompt = "Describe this image in detail for an alt text. Be concise but informative."

        return self.provider.generate_image_description(image_path, prompt)

    def improve_table_conversion(self, html_table: str) -> str:
        """
        Improve table conversion using LLM.

        Parameters
        ----------
        html_table : str
            The HTML table to convert.

        Returns
        -------
        str
            The improved Markdown table.
        """
        prompt = f"""
        Convert this HTML table to a well-formatted GitHub Flavored Markdown table.
        Ensure proper column alignment and formatting.
        DO NOT wrap your response in triple backticks (```) or add markdown code block syntax.
        Return ONLY the formatted markdown table, with no surrounding code block markers.

        HTML Table:
        {html_table}
        """

        system_message = """
        You are a helpful assistant specialized in converting HTML tables to Markdown.
        You create well-formatted GitHub Flavored Markdown tables with proper alignment.
        Include only the markdown table in your response, no explanations or other text.
        IMPORTANT: DO NOT wrap your response in triple backticks (```) or markdown code block syntax.
        """

        # Create parameters with correct token limit format for the model
        text_gen_params = {
            "prompt": prompt,
            "system_message": system_message,
            "max_tokens": 4000,  # Increased for handling larger document structures
        }

        # Only add temperature parameter for models that support it
        if not "o3-mini" in getattr(self.provider, 'model', ''):
            text_gen_params["temperature"] = 0.2  # Lower temperature for more consistent results

        response = self.provider.generate_text(**text_gen_params)

        # Strip any markdown code block markers that might have been added
        return self._strip_markdown_code_markers(response)

    def extract_structure_from_pdf(self, pdf_text: str) -> Dict[str, Any]:
        """
        Extract document structure from PDF text using LLM.

        Parameters
        ----------
        pdf_text : str
            The raw text extracted from a PDF.

        Returns
        -------
        Dict[str, Any]
            The extracted document structure.
        """
        # Sample the text rather than truncating, to get a better overview
        # of the entire document structure
        sample_text = self._create_representative_sample(pdf_text, max_chars=4000)

        prompt = f"""
        Analyze this text extracted from a PDF document and identify its structure.
        Extract the following elements:
        - Title
        - Headings and subheadings (with their levels)
        - Lists (bullet points and numbered)
        - Tables (identify where tables begin and end)
        - Figures/images (identify where they are mentioned)

        Format your response as a JSON object with these keys:
        - title: string
        - headings: array of objects with 'text' and 'level' properties
        - lists: array of objects with 'items' and 'type' (bullet or numbered) properties
        - tables: array of objects with 'start_text' and 'end_text' properties
        - figures: array of strings mentioning figures

        IMPORTANT: Return ONLY the raw JSON with no surrounding code block markers.
        DO NOT wrap your JSON in triple backticks (```) or add any other non-JSON text.

        Here is the PDF text (sampled from the full document):

        {sample_text}
        """

        system_message = """
        You are a document structure analysis expert.
        You can identify the structure of documents from raw text.
        Respond only with a valid JSON object containing the requested information.
        DO NOT wrap your response in code block markers, markdown formatting, or explanatory text.
        Return ONLY the raw JSON object.
        """

        # Create parameters with correct token limit format for the model
        text_gen_params = {
            "prompt": prompt,
            "system_message": system_message,
            "max_tokens": 4000,  # Increased for handling larger structure analysis
        }

        # Only add temperature parameter for models that support it
        if not "o3-mini" in getattr(self.provider, 'model', ''):
            text_gen_params["temperature"] = 0.2  # Lower temperature for more consistent results

        response = self.provider.generate_text(**text_gen_params)

        # Strip any possible code block markers
        response = self._strip_markdown_code_markers(response)

        # Parse the JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, return a basic structure
            return {
                "title": "",
                "headings": [],
                "lists": [],
                "tables": [],
                "figures": []
            }

    def _create_representative_sample(self, text: str, max_chars: int = 4000) -> str:
        """
        Create a representative sample of the text that includes portions from
        beginning, middle, and end of the document.

        Parameters
        ----------
        text : str
            The full text to sample from.
        max_chars : int, optional
            Maximum characters in the sample.

        Returns
        -------
        str
            The representative sample text.
        """
        if len(text) <= max_chars:
            return text

        # Calculate segment sizes
        segment_size = max_chars // 3

        # Get beginning, middle and end segments
        beginning = text[:segment_size]

        middle_start = max(0, len(text)//2 - segment_size//2)
        middle = text[middle_start:middle_start + segment_size]

        end_start = max(0, len(text) - segment_size)
        end = text[end_start:]

        # Combine segments with markers
        sample = (
            beginning +
            "\n\n[...]\n\n" +
            middle +
            "\n\n[...]\n\n" +
            end
        )

        return sample
