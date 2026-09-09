import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Conditional import for the 'markdown' library.
# This ensures that the node can be defined even if the library isn't present
# at import time, but will fail gracefully when `process` is called.
try:
    import markdown
except ImportError:
    logging.getLogger(__name__).warning(
        "The 'markdown' library is not installed. "
        "MarkdownParserNode will not function without it. "
        "Please install with 'pip install markdown'."
    )
    markdown = None  # Set to None so we can check its presence in process()


logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for converting Markdown formatted
    text into HTML.

    This node expects its input `data` to be a string containing Markdown.
    It utilizes the 'markdown' library for the conversion process.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, transforming Markdown text into HTML.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary providing contextual information for the node's
                     operation. Not directly used for the parsing logic in this
                     particular node, but available for potential future extensions
                     or logging.

        Returns:
            A string representing the HTML output derived from the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If the 'markdown' library is not available during
                          processing.
            Exception: Captures and re-raises any unexpected exceptions that
                       occur during the Markdown parsing process.
        """
        logger.debug(f"[{self.node_name}] Starting Markdown parsing for incoming data.")

        if markdown is None:
            logger.critical(
                f"[{self.node_name}] 'markdown' library is not available. "
                "Cannot perform Markdown parsing. Please install it."
            )
            raise RuntimeError(
                f"[{self.node_name}] Required 'markdown' library is not installed."
            )

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {data[:100]}..."
            )
            raise TypeError(
                f"[{self.node_name}] Expected 'data' to be a string for Markdown parsing, "
                f"but received '{type(data).__name__}'."
            )

        try:
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed Markdown data to HTML.")
            return html_output
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing."
            )
            raise RuntimeError(f"[{self.node_name}] Failed to parse Markdown: {e}") from e

