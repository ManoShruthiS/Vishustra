import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing markdown text and
    transforming it into HTML format. This node simulates the core conversion
    logic, applying common markdown-to-HTML rules.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting markdown strings into HTML.
        This method includes basic transformations for common markdown elements
        like bold and italic text, and wraps the content in a paragraph tag
        if it's not already structured.

        Args:
            data: The input data, expected to be a string containing markdown text.
            context: A dictionary containing contextual information relevant to the
                     processing task (currently unused by this specific node).

        Returns:
            A string representing the HTML output parsed from the markdown.

        Raises:
            ValueError: If the input 'data' is not a string.
            RuntimeError: If an unexpected error occurs during the markdown parsing
                          simulation.
        """
        if not isinstance(data, str):
            logger.error(
                "Invalid input type for MarkdownParserNode. Expected string, but received %s.",
                type(data).__name__
            )
            raise ValueError(
                f"MarkdownParserNode expects string data, but received type {type(data).__name__}"
            )

        try:
            # Simulate markdown parsing. In a production environment,
            # this would typically involve an external markdown library
            # (e.g., `markdown`, `mistune`, `commonmark`).
            # For this node, we'll apply some basic regex-based transformations.

            # Convert bold text: **text** -> <strong>text</strong>
            processed_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', data)
            # Convert italic text: *text* -> <em>text</em>
            processed_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_html)

            # Basic paragraph wrapping if the content doesn't appear to be structured HTML already
            stripped_html = processed_html.strip()
            if not stripped_html.startswith('<') or not stripped_html.endswith('>'):
                processed_html = f"<p>{stripped_html}</p>"

            logger.info("Successfully transformed markdown data into HTML format.")
            return processed_html
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during markdown parsing simulation in MarkdownParserNode."
            )
            raise RuntimeError(f"Failed to parse markdown: {e}") from e