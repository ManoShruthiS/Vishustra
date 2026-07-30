import logging
import re
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class NodeProcessingError(Exception):
    """Custom exception for errors encountered during node processing."""
    pass

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse Markdown text and convert common
    Markdown elements into a simplified HTML-like string representation.
    This node serves as an example of text transformation within the Vishustra framework.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, which is expected to be a Markdown string,
        and transforms common Markdown elements into simplified HTML-like tags.

        Args:
            data: The input data, anticipated to be a string containing Markdown.
            context: A dictionary providing runtime context information
                     (e.g., global variables, configuration).

        Returns:
            A string where recognized Markdown elements have been converted
            to simplified HTML-like tags.

        Raises:
            NodeProcessingError: If the input data is not a string, or if an
                                 unexpected issue occurs during the parsing process.
        """
        if not isinstance(data, str):
            logger.error(
                "MarkdownParserNode received invalid input type. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise NodeProcessingError(
                f"Invalid input type for MarkdownParserNode. Expected 'str', got '{type(data).__name__}'."
            )

        logger.info("Starting Markdown parsing for input data (length: %d characters).", len(data))
        processed_content = data

        try:
            # Simulate parsing and conversion of Markdown elements

            # Convert headers (H1, H2, H3)
            processed_content = self._replace_header_tags(processed_content)

            # Convert bold text: **text** -> <strong>text</strong>
            processed_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_content, flags=re.DOTALL)

            # Convert italic text: *text* -> <em>text</em>
            # This regex is intentionally simple and might not handle all edge cases
            # (e.g., asterisks within code blocks) without more sophisticated parsing.
            processed_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_content, flags=re.DOTALL)

            logger.info("Successfully processed Markdown content. Output length: %d characters.", len(processed_content))
            return processed_content
        except Exception as e:
            logger.exception("An unexpected error occurred during Markdown parsing within MarkdownParserNode.")
            raise NodeProcessingError(f"Failed to parse Markdown content: {e}") from e

    def _replace_header_tags(self, text: str) -> str:
        """
        Helper method to replace Markdown header syntax (#, ##, ###) with
        corresponding simplified HTML-like header tags (<h1>, <h2>, <h3>).

        Args:
            text: The input string potentially containing Markdown headers.

        Returns:
            The string with header syntax replaced.
        """
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            if line.startswith('### '):
                processed_lines.append(f"<h3>{line[4:].strip()}</h3>")
            elif line.startswith('## '):
                processed_lines.append(f"<h2>{line[3:].strip()}</h2>")
            elif line.startswith('# '):
                processed_lines.append(f"<h1>{line[2:].strip()}</h1>")
            else:
                processed_lines.append(line)
        return '\n'.join(processed_lines)