import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Attempt to import the markdown library. If not found, log a critical error
# and handle its absence gracefully in the process method.
try:
    import markdown as md
except ImportError:
    # A dedicated logger for this file.
    _initial_logger = logging.getLogger(__name__)
    _initial_logger.critical(
        "The 'markdown' library is not installed. "
        "Please install it using 'pip install markdown' to enable the MarkdownParserNode."
    )
    # Set md to None to indicate the dependency is missing.
    md = None

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node designed to parse Markdown formatted text and
    transform it into HTML.

    This node expects a string input containing valid Markdown syntax.
    It leverages the 'python-markdown' library for efficient and
    standard-compliant Markdown to HTML conversion.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input Markdown string into its HTML representation.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information. This node
                     does not currently utilize the context, but it's available
                     for potential future enhancements like passing markdown
                     extensions.

        Returns:
            A string containing the HTML output derived from the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If the 'markdown' library is not installed or
                          if an unexpected error occurs during the parsing process.
        """
        logger.debug(f"[{self.node_name}] Initiating processing for input data type: {type(data)}")

        # Check if the markdown library was successfully imported.
        if md is None:
            error_msg = (
                f"[{self.node_name}] 'markdown' library is not available. "
                "Please ensure it is installed to use this node."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Validate input data type.
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected a string, "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Perform the Markdown to HTML conversion.
            # In a more advanced scenario, 'context' could pass markdown extensions:
            # e.g., extensions = context.get("markdown_extensions", [])
            # html_output = md.markdown(data, extensions=extensions)
            html_output = md.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed Markdown to HTML.")
            return html_output
        except Exception as e:
            # Catch any exceptions during parsing and log them.
            error_msg = f"[{self.node_name}] Failed to parse Markdown: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e