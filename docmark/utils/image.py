"""
Image utility module for DocMark.

This module provides functionality for processing images, including
extraction, conversion, and metadata handling.
"""

import os
import base64
import hashlib
import tempfile
import mimetypes
from typing import Optional, Dict, Any, Tuple, cast

try:
    from PIL import Image, ExifTags, UnidentifiedImageError
    import exifread
except ImportError as exc:
    raise ImportError("Pillow and exifread are required. Install with 'pip install Pillow exifread'") from exc


class ImageProcessor:
    """Image processing utility class."""

    def __init__(self, images_dir: Optional[str] = None):
        """
        Initialize the image processor.

        Parameters
        ----------
        images_dir : str, optional
            Directory where images will be saved. If None, a temporary directory will be used.
        """
        self.images_dir = images_dir
        self.temp_dir = None

        if not self.images_dir:
            # Create a temporary directory to store images
            self.temp_dir = tempfile.TemporaryDirectory()
            self.images_dir = self.temp_dir.name
        else:
            os.makedirs(self.images_dir, exist_ok=True)

        # Dictionary to track image counters for each document
        self.image_counters: Dict[str, int] = {}

    def __del__(self) -> None:
        """Clean up temporary directory if created."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir:
                self.temp_dir.cleanup()
        except Exception:
            # Silent cleanup error in destructor
            pass

    def extract_image(self, image_data: bytes, content_type: str, prefix: str = "image",
                     doc_filename: Optional[str] = None) -> str:
        """
        Extract and save an image from binary data.

        Parameters
        ----------
        image_data : bytes
            Binary image data.
        content_type : str
            MIME type of the image.
        prefix : str, optional
            Prefix for the image filename.
        doc_filename : str, optional
            Source document filename. If provided, images will be named using
            the document filename + _img1, img2, etc.

        Returns
        -------
        str
            Path to the saved image.
        """
        # Determine file extension from content type
        extension = mimetypes.guess_extension(content_type) or ".png"

        # Check if format needs conversion
        needs_conversion = extension.lower() in [".emf", ".wmf", ".tif", ".tiff", ".bmp"]
        final_extension = ".png" if needs_conversion else extension

        # Generate a hash of the image data for a unique filename (used in both branches)
        image_hash = hashlib.md5(image_data).hexdigest()[:10]

        # Create filename based on document name if provided
        if doc_filename:
            # Extract base filename without extension
            base_filename = os.path.splitext(os.path.basename(doc_filename))[0]

            # Get or initialize counter for this document
            if base_filename not in self.image_counters:
                self.image_counters[base_filename] = 1
            else:
                self.image_counters[base_filename] += 1

            # Create filename with sequential numbering
            image_filename = f"{base_filename}_img{self.image_counters[base_filename]}{final_extension}"
        else:
            # Fallback to hash-based naming if no document filename provided
            image_filename = f"{prefix}_{image_hash}{final_extension}"

        # Ensure images_dir is not None before joining
        images_dir = cast(str, self.images_dir)
        image_path = os.path.join(images_dir, image_filename)

        # Save the image
        if needs_conversion:
            # Save to temporary file first
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_img_{image_hash}{extension}")

            # Use with for resource-allocating operations
            with open(temp_path, "wb") as f:
                f.write(image_data)

            # Convert to PNG
            try:
                self.convert_to_png(temp_path, image_path)
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                # Fall back to original format if conversion fails
                image_filename = f"{prefix}_{image_hash}{extension}"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_data)
        else:
            # Save in original format
            with open(image_path, "wb") as f:
                f.write(image_data)

        return image_path

    def convert_to_png(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert an image to PNG format.

        Parameters
        ----------
        input_path : str
            Path to the input image.
        output_path : str, optional
            Path where the PNG will be saved. If None, creates a temp file.

        Returns
        -------
        str
            Path to the converted PNG file.

        Raises
        ------
        RuntimeError
            If conversion fails.
        """
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name

        try:
            # Convert image using Pillow
            with Image.open(input_path) as img:
                # If it's not RGB (like RGBA or CMYK), convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(output_path, "PNG")
                return output_path

        except (IOError, OSError, ValueError, ImportError, UnidentifiedImageError) as e:
            # Clean up the temp file if conversion failed
            if output_path != input_path and os.path.exists(output_path):
                os.unlink(output_path)

            raise RuntimeError(f"Failed to convert image to PNG: {str(e)}") from e

    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get the dimensions of an image.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        Tuple[int, int]
            Width and height of the image.
        """
        with Image.open(image_path) as img:
            return img.width, img.height

    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an image.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        Dict[str, Any]
            Dictionary of metadata.
        """
        if not os.path.exists(image_path):
            return {}

        metadata: Dict[str, Any] = {}

        try:
            # Extract basic image information with PIL
            with Image.open(image_path) as img:
                metadata["ImageSize"] = f"{img.width}x{img.height}"

                # Extract PIL's Exif data
                if hasattr(img, "getexif") and img.getexif():
                    exif = {
                        ExifTags.TAGS.get(tag, tag): value
                        for tag, value in img.getexif().items()
                        if tag in ExifTags.TAGS
                    }

                    # Map common Exif tags to our metadata fields
                    if "DateTimeOriginal" in exif:
                        metadata["DateTimeOriginal"] = exif["DateTimeOriginal"]
                    elif "DateTime" in exif:
                        metadata["CreateDate"] = exif["DateTime"]

                    if "Artist" in exif:
                        metadata["Artist"] = exif["Artist"]

                    if "ImageDescription" in exif:
                        metadata["Description"] = exif["ImageDescription"]

                    if "UserComment" in exif:
                        if "Description" in metadata:
                            metadata["Caption"] = exif["UserComment"]
                        else:
                            metadata["Description"] = exif["UserComment"]

            # Use ExifRead for more detailed metadata
            with open(image_path, "rb") as f:
                exif_tags = exifread.process_file(f)

                # Extract GPS data if available
                if "GPS GPSLatitude" in exif_tags and "GPS GPSLongitude" in exif_tags:
                    try:
                        lat = self._convert_to_degrees(
                            exif_tags["GPS GPSLatitude"].values
                        )
                        lon = self._convert_to_degrees(
                            exif_tags["GPS GPSLongitude"].values
                        )

                        # Apply negative sign for South and West
                        if (
                            "GPS GPSLatitudeRef" in exif_tags
                            and exif_tags["GPS GPSLatitudeRef"].values == "S"
                        ):
                            lat = -lat
                        if (
                            "GPS GPSLongitudeRef" in exif_tags
                            and exif_tags["GPS GPSLongitudeRef"].values == "W"
                        ):
                            lon = -lon

                        metadata["GPSPosition"] = f"{lat:.6f}, {lon:.6f}"
                    except (ValueError, TypeError, AttributeError):
                        pass

                # Extract keywords/tags if available
                if "Image Keywords" in exif_tags:
                    keywords = str(exif_tags["Image Keywords"].values)
                    if keywords:
                        metadata["Keywords"] = keywords

                # Get title if available
                if "Image DocumentName" in exif_tags:
                    metadata["Title"] = str(exif_tags["Image DocumentName"].values)

            return metadata

        except (
            IOError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            UnidentifiedImageError,
        ):
            # If any error occurs during metadata extraction, return empty dict
            return {}

    def _convert_to_degrees(self, values: Any) -> float:
        """
        Helper method to convert GPS coordinates from degree/minute/second format to decimal degrees.
        """
        degrees = float(values[0].num) / float(values[0].den)
        minutes = float(values[1].num) / float(values[1].den)
        seconds = float(values[2].num) / float(values[2].den)

        return degrees + (minutes / 60.0) + (seconds / 3600.0)

    def resize_image(
        self, image_path: str, max_width: int = 800, max_height: int = 600
    ) -> str:
        """
        Resize an image while maintaining aspect ratio.

        Parameters
        ----------
        image_path : str
            Path to the image.
        max_width : int, optional
            Maximum width of the resized image.
        max_height : int, optional
            Maximum height of the resized image.

        Returns
        -------
        str
            Path to the resized image.
        """
        with Image.open(image_path) as img:
            # Get original dimensions
            width, height = img.size

            # Calculate aspect ratio
            aspect_ratio = width / height

            # Calculate new dimensions
            if width > max_width or height > max_height:
                if width / max_width > height / max_height:
                    # Width is the limiting factor
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Height is the limiting factor
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)

                # Resize the image with high quality
                # Use a simple integer for resampling to avoid version-specific constants
                # 1 = NEAREST, 2 = BOX, 3 = BILINEAR, 4 = HAMMING, 5 = BICUBIC, 6 = LANCZOS
                resized_img = img.resize((new_width, new_height), 5)  # BICUBIC is widely supported

                # Create output path
                filename, ext = os.path.splitext(image_path)
                output_path = f"{filename}_resized{ext}"

                # Save the resized image
                resized_img.save(output_path)

                return output_path

            # If no resizing needed, return original path
            return image_path

    def image_to_base64(self, image_path: str) -> str:
        """
        Convert an image to base64 encoding.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        str
            Base64-encoded image data.
        """
        with open(image_path, "rb") as image_file:
            content_type, _ = mimetypes.guess_type(image_path)
            content_type = content_type or "image/jpeg"
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            return f"data:{content_type};base64,{base64_image}"

    def get_relative_path(self, image_path: str, base_path: str) -> str:
        """
        Get the relative path of an image from a base path.

        Parameters
        ----------
        image_path : str
            Path to the image.
        base_path : str
            Base path for the relative path.

        Returns
        -------
        str
            Relative path of the image.
        """
        # Get the directory of the base path
        base_dir = os.path.dirname(base_path)

        # Normalize paths to handle case sensitivity and different path formats
        image_path_norm = os.path.normpath(os.path.abspath(image_path))
        base_dir_norm = os.path.normpath(os.path.abspath(base_dir))

        try:
            # Try to get a simple relative path
            rel_path = os.path.relpath(image_path_norm, base_dir_norm)
            rel_path = rel_path.replace("\\", "/")

            # Ensure path starts with ./ if it's not an absolute path
            if not rel_path.startswith("/") and not rel_path.startswith("./"):
                rel_path = "./" + rel_path

            return rel_path
        except ValueError:
            # If paths are on different drives, use the media directory name only
            media_dir = os.path.basename(os.path.dirname(image_path))
            image_name = os.path.basename(image_path)
            return f"./{media_dir}/{image_name}"
