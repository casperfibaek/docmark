"""
Command-line interface for DocMark.

This module provides a command-line interface for the DocMark library.
"""

import sys
from typing import Optional
import click

from docmark import convert as docmark_convert, batch_convert


@click.group()
@click.version_option()
def cli():
    """DocMark: Convert between Word files and Markdown."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False)
@click.option(
    "--from",
    "from_format",
    type=click.Choice(["docx", "md", "pdf", "html"]),
    help="Source format. If not specified, will be derived from input file extension.",
)
@click.option(
    "--to",
    "to_format",
    type=click.Choice(["docx", "md"]),
    help="Target format. If not specified, will be derived from output file extension.",
)
@click.option(
    "--images-dir",
    type=click.Path(),
    help="Directory for extracted images. If not specified, will use 'media' folder.",
)
@click.option(
    "--template",
    type=click.Path(exists=True),
    help="Path to the template file for DOCX output.",
)
@click.option(
    "--toc/--no-toc",
    default=False,
    help="Include table of contents in DOCX output.",
)
@click.option(
    "--api-key",
    help="API key for the LLM provider. If not specified, will try to get from environment.",
)
@click.option(
    "--model",
    help="Model to use for LLM requests. If not specified, will use a default model.",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider to use.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Disable LLM usage for formatting enhancement.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print verbose output.",
)
def convert(
    input_path: str,
    output_path: Optional[str],
    from_format: Optional[str],
    to_format: Optional[str],
    images_dir: Optional[str],
    template: Optional[str],
    toc: bool,
    api_key: Optional[str],
    model: Optional[str],
    provider: str,
    no_llm: bool,
    verbose: bool,
):
    """
    Convert a document from one format to another.

    INPUT_PATH is the path to the input file.
    OUTPUT_PATH is the path to the output file. If not specified, will be derived from INPUT_PATH.
    """
    try:
        # Convert the document
        output_file = docmark_convert(
            input_path=input_path,
            output_path=output_path,
            llm_api_key=api_key,
            llm_model=model,
            llm_provider=provider,
            verbose=verbose,
            from_format=from_format,
            to_format=to_format,
            images_dir=images_dir,
            template_path=template,
            toc=toc,
            use_llm=not no_llm,
        )

        click.echo(f"Conversion completed: {output_file}")
        return 0
    except (ValueError, IOError, FileNotFoundError) as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False), required=False)
@click.option(
    "--pattern",
    default="*.*",
    help="Pattern to match input files.",
)
@click.option(
    "--from",
    "from_format",
    type=click.Choice(["docx", "md", "pdf", "html"]),
    help="Source format. If not specified, will be derived from input file extension.",
)
@click.option(
    "--to",
    "to_format",
    type=click.Choice(["docx", "md"]),
    help="Target format. If not specified, will be derived from output file extension.",
)
@click.option(
    "--images-dir",
    type=click.Path(),
    help="Directory for extracted images. If not specified, will use 'media' folder.",
)
@click.option(
    "--template",
    type=click.Path(exists=True),
    help="Path to the template file for DOCX output.",
)
@click.option(
    "--toc/--no-toc",
    default=False,
    help="Include table of contents in DOCX output.",
)
@click.option(
    "--api-key",
    help="API key for the LLM provider. If not specified, will try to get from environment.",
)
@click.option(
    "--model",
    help="Model to use for LLM requests. If not specified, will use a default model.",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider to use.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Disable LLM usage for formatting enhancement.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print verbose output.",
)
def batch(
    input_dir: str,
    output_dir: Optional[str],
    pattern: str,
    from_format: Optional[str],
    to_format: Optional[str],
    images_dir: Optional[str],
    template: Optional[str],
    toc: bool,
    api_key: Optional[str],
    model: Optional[str],
    provider: str,
    no_llm: bool,
    verbose: bool,
):
    """
    Convert multiple documents in a directory.

    INPUT_DIR is the directory containing input files.
    OUTPUT_DIR is the directory for output files. If not specified, will use INPUT_DIR.
    """
    try:
        # Convert the documents
        output_files = batch_convert(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            llm_api_key=api_key,
            llm_model=model,
            llm_provider=provider,
            verbose=verbose,
            from_format=from_format,
            to_format=to_format,
            images_dir=images_dir,
            template_path=template,
            toc=toc,
            use_llm=not no_llm,
        )

        if output_files:
            click.echo(f"Converted {len(output_files)} files:")
            for output_file in output_files:
                click.echo(f"  {output_file}")
        else:
            click.echo("No files were converted.")
        return 0
    except (ValueError, IOError, FileNotFoundError) as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


def main():
    """Run the CLI."""
    return cli()


if __name__ == "__main__":
    sys.exit(main())
