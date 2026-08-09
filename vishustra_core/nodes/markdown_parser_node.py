import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Try to import the 'markdown' library.
# This check is performed once when the module is loaded, optimizing performance
# and allowing early detection of missing critical dependencies.
try:
    import markdown
    _MARKDOWN_LIB_AVAILABLE = True
    logger.debug("Successfully imported 'markdown' library for MarkdownParserNode.")
except ImportError:
    _MARKDOWN_LIB_AVAILABLE = False
    logger.critical(
        "The 'markdown' library is not installed. MarkdownParserNode will "
        "raise a RuntimeError if used without this critical dependency. "
        "Please install it with 'pip install markdown'."
    )

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to convert Markdown text into HTML.

    This node expects a string containing valid Markdown as its primary input
    and produces an HTML string as its output. It integrates with the
    'markdown' Python library to provide robust and feature-rich parsing.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses Markdown input data and returns the resulting HTML string.

        Args:
            data: The input data, which is expected to be a string containing
                  Markdown content.
            context: A dictionary containing contextual information relevant
                     to the processing flow. This node currently does not
                     utilize the context dictionary but adheres to the interface.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string, indicating an
                       incorrect data type for Markdown parsing.
            RuntimeError: If the 'markdown' library is not installed, which
                          is a critical dependency for this node's core
                          functionality.
            Exception: Propagates any other exceptions that may occur during
                       the Markdown parsing process after logging them,
                       allowing upstream error handling.
        """
        logger.debug(f"[{self.node_name}] Starting process. Input data type: {type(data).__name__}.")

        # Validate input data type to ensure it's a string
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'.")
            raise TypeError(
                f"MarkdownParserNode expects input data to be a string, "
                f"but received '{type(data).__name__}'.")

        # Check for the availability of the 'markdown' library
        if not _MARKDOWN_LIB_AVAILABLE:
            # If the library is missing, log a critical error and raise an exception
            # as the node cannot fulfill its primary function.
            logger.critical(
                f"[{self.node_name}] Critical dependency 'markdown' is not installed. "
                f"Cannot perform Markdown parsing. Please install with 'pip install markdown'.")
            raise RuntimeError(
                f"MarkdownParserNode requires the 'markdown' library, which is not installed. "
                f"Please install it with 'pip install markdown'.")

        try:
            # Perform the Markdown to HTML conversion using the imported library
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed Markdown data to HTML.")
            return html_output
        except Exception as e:
            # Catch and log any unexpected errors during the parsing process
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}",
                exc_info=True  # Include traceback for detailed debugging
            )
            # Re-raise the exception to allow upstream nodes or the orchestration
            # framework to handle the failure appropriately.
            raise