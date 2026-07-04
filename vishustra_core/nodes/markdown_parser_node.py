import logging
from typing import Any, Dict

import markdown # Third-party library for markdown parsing

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node expects a string containing Markdown as input and converts it
    to its corresponding HTML representation using the 'markdown' library.
    It can optionally accept a list of Markdown extensions via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input Markdown string and returns the resulting HTML.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary for contextual information.
                                      Can include 'markdown_extensions' (list of str)
                                      to enable specific Markdown extensions.
                                      e.g., `{"markdown_extensions": ["fenced_code"]}`

        Returns:
            Any: The processed data, typically an HTML string.

        Raises:
            TypeError: If the input data is not a string.
            RuntimeError: If an error occurs during Markdown parsing.
        """
        if not isinstance(data, str):
            logger.error(
                f"{self.node_name}: Invalid input data type. "
                f"Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"Input data for MarkdownParserNode must be a string. "
                f"Got {type(data).__name__}."
            )

        try:
            # Retrieve optional markdown extensions from context
            extensions = context.get("markdown_extensions", [])
            if not isinstance(extensions, list):
                logger.warning(
                    f"{self.node_name}: 'markdown_extensions' in context "
                    f"is not a list. Ignoring provided extensions."
                )
                extensions = []

            # Perform the markdown parsing
            processed_data = markdown.markdown(data, extensions=extensions)

            logger.info(f"{self.node_name}: Successfully parsed markdown to HTML.")
            return processed_data

        except Exception as e:
            logger.error(
                f"{self.node_name}: An unexpected error occurred during markdown parsing: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to parse markdown content: {e}") from e