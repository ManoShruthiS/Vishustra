import logging
from typing import Any, Dict, Union
import markdown

# Assuming BaseNode is available at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node expects a string containing Markdown formatted text as input
    and returns a string containing the corresponding HTML.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Parses the input data from Markdown to HTML.

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary containing contextual information for processing.
                     (Not directly used by this specific node, but required by BaseNode interface).

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            ValueError: If the input data is not a string.
            Exception: For any unexpected errors during the Markdown parsing process.
        """
        if not isinstance(data, str):
            error_msg = f"MarkdownParserNode expects a string input, but received type: {type(data).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not data.strip():
            logger.warning("MarkdownParserNode received an empty or whitespace-only string for parsing.")
            return ""

        try:
            # Use the markdown library to convert markdown to HTML
            html_output = markdown.markdown(data)
            logger.debug("Successfully parsed markdown content to HTML.")
            return html_output
        except Exception as e:
            error_msg = f"An unexpected error occurred during Markdown parsing: {e}"
            logger.exception(error_msg) # Logs the full traceback
            raise RuntimeError(error_msg) from e
