import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Ensure the 'markdown' library is installed: pip install markdown
import markdown

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the 'markdown' Python library to convert a given
    Markdown string into its corresponding HTML representation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, parsing it as Markdown and returning the HTML output.

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary providing contextual information for processing.
                     Not directly used by this node but available for future extensions.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If an error occurs during the markdown parsing process.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected string, but received {type(data).__name__}.")
            raise TypeError(f"'{self.node_name}' expects a string input for markdown parsing, received {type(data).__name__}.")

        logger.debug(f"[{self.node_name}] Starting markdown parsing for input data (length: {len(data)} characters).")

        try:
            # The markdown library converts markdown string to HTML string
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed markdown to HTML. Output length: {len(html_output)} characters.")
            return html_output
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during markdown parsing.")
            raise RuntimeError(f"Failed to parse markdown content: {e}") from e

