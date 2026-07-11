import logging
import re
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
# For local testing, you might need to adjust this import path or create a dummy base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that parses markdown-formatted text and converts it
    into a simplified HTML representation. This node aims to simulate common
    markdown-to-HTML transformations.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, expecting a markdown string, and transforms it
        into a simplified HTML string. The transformation covers basic elements
        like headers, bold, italic, and links.

        Args:
            data: The input data, expected to be a string containing markdown.
            context: A dictionary of contextual information. This node does not
                     directly utilize the context for its core parsing logic but
                     it is available for potential future extensions or logging.

        Returns:
            A string containing the HTML representation of the markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If parsing encounters an unexpected issue during the
                        transformation process.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for MarkdownParserNode. Expected string, "
                f"got {type(data).__name__}."
            )
            raise TypeError(
                f"MarkdownParserNode expects string input, but received {type(data).__name__}."
            )

        if not data.strip():
            logger.debug("Received empty or whitespace-only markdown string. Returning empty string.")
            return ""

        parsed_text = data

        try:
            # Convert Headers: # H1, ## H2, etc. up to H6
            # Regex captures the number of '#' characters and the header text.
            parsed_text = re.sub(
                r'^(#{1,6})\s*(.*)$',
                lambda m: f'<h{len(m.group(1))}>{m.group(2).strip()}</h{len(m.group(1))}>',
                parsed_text,
                flags=re.MULTILINE
            )
            
            # Convert Bold: **text** or __text__
            # This regex handles both patterns by putting the content in different groups.
            # \1\2 ensures that whichever group matched (or both if overlapping, but that's
            # less common for simple bold), its content is used.
            parsed_text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<strong>\1\2</strong>', parsed_text)
            
            # Convert Italic: *text* or _text_
            # Similar logic to bold.
            parsed_text = re.sub(r'\*(.*?)\*|_(.*?)_', r'<em>\1\2</em>', parsed_text)

            # Convert Links: [text](url)
            # Captures link text (\1) and URL (\2).
            parsed_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', parsed_text)
            
            # Basic Paragraph handling: Wrap lines that are not already block elements.
            # This is a simplification; a full parser would analyze block context.
            lines = parsed_text.split('\n')
            processed_lines = []
            for line in lines:
                stripped_line = line.strip()
                # Check if the line is not empty and doesn't start with a known HTML block tag
                # (h1-h6, strong, em, a, p - though p might be what we're adding).
                # This prevents double-wrapping already converted elements.
                if stripped_line and not re.match(r'<(h[1-6]|strong|em|a|p|ul|ol|li|div|table|blockquote)', stripped_line, re.IGNORECASE):
                    processed_lines.append(f"<p>{stripped_line}</p>")
                else:
                    processed_lines.append(line)
            parsed_text = "\n".join(processed_lines)

            logger.info("Successfully parsed markdown string into simplified HTML.")
            return parsed_text
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during markdown parsing in MarkdownParserNode: {e}",
                exc_info=True
            )
            raise ValueError(
                f"Failed to parse markdown due to an internal error: {e}. "
                "Check logs for more details."
            ) from e
