
import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node that parses Markdown formatted text and transforms it
    into a simplified HTML string. This node is useful for preparing
    user-friendly content or for further processing where HTML is preferred.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts
        it into a simplified HTML string.

        Supports basic Markdown elements:
        - Headers (#, ##, ###, ...)
        - Bold (**, __)
        - Italic (*, _)
        - Paragraphs (newline separated)

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary of contextual information for the processing.

        Returns:
            A string representing the simplified HTML output.

        Raises:
            TypeError: If the input data is not a string.
            Exception: For other unexpected errors during parsing.
        """
        if not isinstance(data, str):
            error_msg = f"MarkdownParserNode received non-string data: {type(data).__name__}. Expected a string."
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"[{self.node_name}] Starting markdown parsing for input data.")

        parsed_html_lines = []
        try:
            # Split into lines to handle block-level elements like headers
            lines = data.split('\n')
            for line in lines:
                stripped_line = line.strip()

                if not stripped_line:
                    continue # Skip empty lines

                # Headers (e.g., # H1, ## H2)
                header_match = re.match(r"^(#+)\s*(.*)", stripped_line)
                if header_match:
                    level = len(header_match.group(1))
                    content = header_match.group(2).strip()
                    parsed_html_lines.append(f"<h{level}>{content}</h{level}>")
                    continue

                # Apply inline formatting for paragraphs
                # Bold: **text** or __text__
                line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                line = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', line)
                
                # Italic: *text* or _text_
                line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line)
                line = re.sub(r'_([^_]+)_', r'<em>\1</em>', line)

                # Wrap remaining lines as paragraphs
                parsed_html_lines.append(f"<p>{line}</p>")

            result = "\n".join(parsed_html_lines)
            logger.info(f"[{self.node_name}] Successfully parsed markdown data.")
            return result

        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during markdown parsing: {e}"
            logger.exception(error_msg)
            raise Exception(error_msg)

