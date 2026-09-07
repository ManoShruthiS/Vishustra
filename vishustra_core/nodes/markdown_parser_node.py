import logging
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Attempt to import the external 'markdown' library
try:
    import markdown
except ImportError:
    logger.critical(
        "The 'markdown' library is not installed. "
        "Please install it using 'pip install markdown' to use MarkdownParserNode."
    )
    # Set markdown to None so we can check its availability during initialization/processing
    markdown = None

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the 'markdown' Python library to perform the conversion.
    It can be configured with various Markdown extensions during initialization.

    If the 'markdown' library is not installed, the node will raise a RuntimeError
    upon instantiation.
    """

    def __init__(self, extensions: Optional[List[str]] = None):
        """
        Initializes the MarkdownParserNode.

        Args:
            extensions: A list of Markdown extension names (e.g., ['fenced_code', 'tables']).
                        These are passed directly to the 'markdown' library.
                        See the 'python-markdown' documentation for available extensions.

        Raises:
            RuntimeError: If the 'markdown' library is not found.
        """
        if markdown is None:
            raise RuntimeError(
                "MarkdownParserNode cannot be initialized: "
                "The 'markdown' library is not installed. "
                "Please install it using 'pip install markdown'."
            )
        self._extensions = extensions if extensions is not None else []
        logger.debug(f"[{self.node_name}] Initialized with extensions: {self._extensions}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Parses the input Markdown string into an HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary containing contextual information. This node does
                     not currently use the context for processing.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            Exception: Catches and re-raises any underlying exceptions from the
                       'markdown' library during parsing, providing additional context.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type for 'data'. "
                f"Expected a string (Markdown text), but received type {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Perform the Markdown to HTML conversion
            html_output = markdown.markdown(data, extensions=self._extensions)
            logger.info(f"[{self.node_name}] Successfully parsed Markdown data to HTML.")
            return html_output
        except Exception as e:
            error_msg = (
                f"[{self.node_name}] An error occurred during Markdown parsing. "
                f"Details: {e}"
            )
            logger.error(error_msg, exc_info=True)
            # Re-raise the exception to propagate the error upstream
            raise
