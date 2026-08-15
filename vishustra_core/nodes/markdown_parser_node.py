import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # Log a critical error if the 'markdown' library is not available.
    # This node requires it for its core functionality.
    logging.critical("The 'markdown' library is not installed. Please install it using 'pip install markdown'.")
    markdown = None  # Set to None to indicate missing dependency

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that takes Markdown formatted text as input
    and converts it into HTML. It relies on the 'markdown' Python library.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting a Markdown string into an HTML string.

        Args:
            data (Any): The input data, expected to be a string containing Markdown content.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. This node does not
                                       directly use the context, but it's part of
                                       the `BaseNode` signature.

        Returns:
            Any: The resulting HTML string after Markdown parsing.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If the 'markdown' library is not available at processing time,
                          or if an unexpected error occurs during parsing.
        """
        if markdown is None:
            logger.error("MarkdownParserNode failed to process: 'markdown' library is missing.")
            raise RuntimeError(
                "Cannot process data. The 'markdown' library is required but not installed. "
                "Please install it using 'pip install markdown'."
            )

        if not isinstance(data, str):
            logger.error(
                f"MarkdownParserNode received invalid data type. Expected str, "
                f"but got {type(data).__name__}."
            )
            raise TypeError(
                f"MarkdownParserNode expects string data for parsing, "
                f"but received type {type(data).__name__}."
            )

        logger.debug(f"[{self.node_name}] Starting Markdown to HTML conversion for input of length {len(data)}.")

        try:
            # The 'markdown' library provides a straightforward function for conversion.
            # Additional extensions could be configured via the 'extensions' argument
            # if specific Markdown flavors were needed (e.g., 'fenced_code', 'tables').
            parsed_html = markdown.markdown(data)
            logger.debug(
                f"[{self.node_name}] Successfully converted Markdown to HTML. "
                f"Output length: {len(parsed_html)}."
            )
            return parsed_html
        except Exception as e:
            # Catching a broad exception to ensure robustness against unexpected issues
            # from the markdown library or underlying system.
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}",
                exc_info=True  # Log full traceback for debugging
            )
            raise RuntimeError(f"Failed to parse Markdown data in {self.node_name}: {e}") from e