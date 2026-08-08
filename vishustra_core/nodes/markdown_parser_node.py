import logging
import re
from typing import Any, Dict

# Assuming 'vishustra_core' is an installed package or part of the project's PYTHONPATH
# and BaseNode is defined within 'nodes/base_node.py' relative to vishustra_core.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node that parses Markdown formatted text into a simpler,
    potentially HTML-like, representation.

    This node simulates parsing common Markdown elements like headers,
    bold, and italics using regular expressions. In a production scenario,
    a dedicated Markdown parsing library would be used.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode.
        No special configuration is needed for this simulated parser.
        """
        logger.debug("MarkdownParserNode initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "Markdown Parser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and returns
        a transformed string with basic Markdown elements converted.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for processing.

        Returns:
            A string with simulated Markdown parsing applied.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If an unexpected error occurs during the parsing process.
        """
        logger.info(f"[{self.node_name}] Starting markdown parsing process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for Markdown parsing."
            )

        # Start with a mutable copy of the input string for transformation
        processed_data = str(data)

        try:
            # --- Simulate Markdown Parsing ---
            # This is a simplified simulation using regex.
            # A real-world implementation would leverage a robust Markdown library
            # like 'markdown', 'mistune', or 'commonmark-py'.

            # Convert Headers (e.g., # Heading -> <h1>Heading</h1>)
            # Matches one or more '#' at the start of a line, followed by space and text.
            processed_data = re.sub(
                r'^(#+)\s*(.*)$',
                lambda m: f'<h{len(m.group(1))}>{m.group(2).strip()}</h{len(m.group(1))}>',
                processed_data,
                flags=re.MULTILINE
            )

            # Convert Bold text (e.g., **text** -> <b>text</b>)
            processed_data = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', processed_data)

            # Convert Italic text (e.g., *text* -> <i>text</i>)
            # This regex is simple and might catch other asterisks; a full parser is more context-aware.
            processed_data = re.sub(r'\*(.*?)\*', r'<i>\1</i>', processed_data)

            # Further parsing logic could be added here for lists, links, code blocks, etc.

            logger.info(f"[{self.node_name}] Successfully parsed markdown data.")
            return processed_data

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during markdown parsing."
            )
            raise RuntimeError(f"[{self.node_name}] Failed to parse markdown data.") from e