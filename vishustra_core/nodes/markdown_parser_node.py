import logging
from typing import Any, Dict

# Assuming this import path is correctly set up within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize a flag to track the availability of the 'markdown' library
_markdown_available = False
try:
    import markdown
    _markdown_available = True
except ImportError:
    # Log a warning if the 'markdown' library is not found.
    # This allows the module to be imported, but the node will raise an error if used.
    logging.getLogger(__name__).warning(
        "The 'markdown' library is not installed. MarkdownParserNode will not be functional. "
        "Please install it using 'pip install markdown' if you intend to use this node."
    )

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that converts Markdown formatted text into HTML.

    This node leverages the standard 'markdown' Python library for reliable and flexible
    Markdown-to-HTML transformation, supporting various Markdown extensions for enhanced
    formatting capabilities. It's designed to take a Markdown string as input and
    produce a corresponding HTML string, suitable for display or further processing.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to HTML.

        The `data` input is strictly expected to be a string containing Markdown.
        The `context` dictionary can optionally include 'markdown_extensions'
        (a list of strings) to customize the parsing behavior, allowing support
        for features like tables, Fenced Code Blocks, etc.

        Args:
            data (Any): The input data, expected to be a Markdown string.
            context (Dict[str, Any]): A dictionary containing context-specific
                                       information. It can optionally contain:
                                       - 'markdown_extensions' (list[str]): A list of
                                         extension names to enable during parsing.

        Returns:
            Any: The processed data as an HTML string.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If the 'markdown' library is not installed or
                          an unexpected error occurs during the parsing process.
        """
        if not _markdown_available:
            logger.error("MarkdownParserNode cannot process data: The required 'markdown' library is not installed.")
            raise RuntimeError("Required 'markdown' library is not installed. Please install it to use MarkdownParserNode.")

        if not isinstance(data, str):
            logger.error(
                f"MarkdownParserNode received unsupported data type: {type(data).__name__}. Expected a string."
            )
            raise TypeError(
                f"MarkdownParserNode expects input data to be a string, "
                f"but received {type(data).__name__}."
            )

        logger.debug(f"MarkdownParserNode: Starting markdown parsing for input of length {len(data)} characters.")

        try:
            # Extract markdown extensions from context if provided
            extensions = context.get('markdown_extensions', [])

            # Validate the type of markdown_extensions from context
            if not isinstance(extensions, list):
                logger.warning(
                    f"Context 'markdown_extensions' is not a list ({type(extensions).__name__}). "
                    "Ignoring invalid extensions and using default parsing."
                )
                extensions = []
            elif not all(isinstance(ext, str) for ext in extensions):
                 logger.warning(
                    f"Context 'markdown_extensions' contains non-string elements. "
                    "Filtering out invalid extensions and using only valid string-based ones."
                )
                 extensions = [ext for ext in extensions if isinstance(ext, str)]

            # Perform the Markdown to HTML conversion
            html_output = markdown.markdown(data, extensions=extensions)
            logger.info("MarkdownParserNode: Successfully converted markdown to HTML.")
            return html_output
        except Exception as e:
            # Catching a broad exception to ensure robustness against unexpected issues
            logger.exception(f"MarkdownParserNode: An unexpected error occurred during markdown parsing: {e}")
            raise RuntimeError(f"Failed to parse markdown content: {e}") from e

