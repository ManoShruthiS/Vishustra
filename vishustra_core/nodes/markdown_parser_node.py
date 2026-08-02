import logging
import re
from typing import Any, Dict

# Assuming this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse markdown formatted text into basic HTML.
    It provides a simple conversion for common markdown elements like bold and italic.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data, expecting a markdown string, and returns its HTML representation.

        Args:
            data: The input markdown string to be parsed.
            context: A dictionary containing contextual information for the node's operation.

        Returns:
            A string representing the HTML equivalent of the input markdown.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        logger.info(f"[{self.node_name}] Starting markdown parsing process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise ValueError(f"{self.node_name} expects string input for markdown parsing.")

        markdown_text: str = data
        parsed_html: str = markdown_text

        try:
            # Simulate basic markdown to HTML conversion
            # Convert **bold** to <strong>bold</strong>
            parsed_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', parsed_html)

            # Convert *italic* to <em>italic</em>
            parsed_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', parsed_html)

            # Add more parsing rules here as needed, e.g., for links, headers, etc.
            # For this initial version, we keep it simple to demonstrate node functionality.

            logger.info(f"[{self.node_name}] Successfully parsed markdown content into HTML.")
            return parsed_html
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during markdown parsing.")
            raise RuntimeError(f"Failed to parse markdown in {self.node_name}: {e}") from e