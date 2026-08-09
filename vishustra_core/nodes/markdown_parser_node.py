import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown text and transform
    it into a simplified HTML string.

    This node demonstrates text transformation by converting common Markdown
    elements such as headers, bold text, and italic text into their
    corresponding HTML tags. It's suitable for scenarios where raw Markdown
    input needs to be presented in a web-friendly or otherwise structured
    HTML format.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "markdown_parser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts
        it into a simplified HTML string.

        The transformation includes:
        - Markdown headers (`#`, `##`, `###`) to `<h1>`, `<h2>`, `<h3>`.
        - Bold text (`**text**`) to `<strong>text</strong>`.
        - Italic text (`*text*`) to `<em>text</em>`.
        - Simple newlines to `<br>` tags for basic line breaks (not full paragraph handling).

        Args:
            data: The input data, which must be a string containing Markdown content.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run (not directly used by this parser
                     but available for future enhancements).

        Returns:
            A string representing the transformed Markdown content in a simplified HTML format.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an unexpected error occurs during the parsing process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected string, "
                f"got {type(data).__name__}. Input data was: {data!r}"
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string (Markdown content)."
            )

        if not data.strip():
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only Markdown content. "
                "Returning an empty string as output."
            )
            return ""

        try:
            processed_html = data

            # Convert Markdown headers (order matters: H3 before H2 before H1)
            processed_html = re.sub(r'###\s*(.*)', r'<h3>\1</h3>', processed_html)
            processed_html = re.sub(r'##\s*(.*)', r'<h2>\1</h2>', processed_html)
            processed_html = re.sub(r'#\s*(.*)', r'<h1>\1</h1>', processed_html)

            # Convert bold text (**text**)
            processed_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_html)

            # Convert italic text (*text*)
            processed_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_html)

            # Replace newlines with <br> tags. This is a very simplistic approach
            # and does not implement proper Markdown paragraph handling.
            # It avoids adding <br> immediately after block-level tags.
            processed_html = re.sub(
                r'\n(?!\s*</?(h[1-6]|ul|ol|li|p|div|pre|blockquote|table|tr|td|th)>|\s*\n)',
                r'<br>\n', processed_html
            )


            logger.info(
                f"[{self.node_name}] Successfully parsed Markdown content."
            )
            return processed_html
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing."
            )
            # Re-raise as a ValueError to indicate a processing failure that
            # upstream nodes might need to handle.
            raise ValueError(f"Failed to parse Markdown content in '{self.node_name}': {e}")