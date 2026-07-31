import logging
from typing import Any, Dict

# Assuming the base_node is part of the vishustra_core package structure
from vishustra_core.nodes.base_node import BaseNode

# Attempt to import the 'markdown' library, which is a required dependency.
try:
    import markdown
except ImportError:
    # Log a critical error and re-raise if the 'markdown' library is not found.
    # This provides immediate feedback on missing dependencies.
    logging.getLogger(__name__).critical(
        "The 'markdown' library is not installed. Please install it using 'pip install markdown'."
    )
    raise

# Initialize a logger for this module to capture events and errors.
logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown content
    and converting it into an HTML string.

    This node expects the input `data` to be a string containing Markdown
    syntax. It leverages the standard 'markdown' Python library for
    robust conversion.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, converting Markdown content into an HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown.
                  If `data` is not a string, a TypeError will be raised.
            context: A dictionary containing runtime context information.
                     This node does not directly utilize the context, but it's
                     part of the BaseNode interface.

        Returns:
            A string representing the HTML output generated from the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during the Markdown
                          parsing process, encapsulating the underlying exception.
        """
        # Validate input data type to ensure it's a string.
        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Invalid input type. Expected a string "
                f"containing Markdown, but received '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        logger.debug(
            f"[{self.node_name}] Starting Markdown to HTML conversion for input "
            f"(first 100 chars): '{data[:100].replace('\\n', ' ').strip()}'..."
        )

        try:
            # Perform the Markdown to HTML conversion.
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully converted Markdown to HTML.")
            return html_output
        except Exception as e:
            # Catch any exceptions during parsing, log them with a traceback,
            # and re-raise as a RuntimeError for consistent error propagation.
            error_message = (
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}"
            )
            logger.exception(error_message)  # Logs the traceback automatically
            raise RuntimeError(f"Failed to parse Markdown content: {e}") from e

