
import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that simulates parsing Markdown text into HTML.
    This node provides basic transformation for common Markdown elements
    such as headers, bold text, italic text, and simple unordered lists.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the input data, which is expected to be a Markdown string,
        and converts it into a simplified HTML string.

        Args:
            data: The Markdown string content to be parsed.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context.

        Returns:
            A dictionary with the key 'html_content' containing the parsed HTML string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an unexpected error occurs during the parsing process.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input type for MarkdownParserNode. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"MarkdownParserNode expects 'data' to be a string, but received {type(data).__name__}.")

        logger.debug(f"Starting Markdown parsing for input of length {len(data)}.")

        parsed_html_blocks = []
        lines = data.strip().split('\n')
        in_list_block = False

        try:
            for line in lines:
                stripped_line = line.strip()

                if not stripped_line:
                    if in_list_block:
                        parsed_html_blocks.append('</ul>')
                        in_list_block = False
                    continue # Skip empty lines

                # Header parsing (H1-H6)
                header_match = re.match(r"^(#+)\s*(.*)", stripped_line)
                if header_match:
                    if in_list_block:
                        parsed_html_blocks.append('</ul>')
                        in_list_block = False
                    level = len(header_match.group(1))
                    content = self._apply_inline_formatting(header_match.group(2).strip())
                    parsed_html_blocks.append(f"<h{level}>{content}</h{level}>")
                    continue

                # Unordered list item parsing (simple: *, -, +)
                list_item_match = re.match(r"^(\s*)[*-]\s+(.*)", stripped_line)
                if list_item_match:
                    if not in_list_block:
                        parsed_html_blocks.append('<ul>')
                        in_list_block = True
                    content = self._apply_inline_formatting(list_item_match.group(2).strip())
                    parsed_html_blocks.append(f"<li>{content}</li>")
                    continue
                else:
                    # If we were in a list block and the current line is not a list item, close the list
                    if in_list_block:
                        parsed_html_blocks.append('</ul>')
                        in_list_block = False

                # Paragraph and general text processing
                processed_line = self._apply_inline_formatting(stripped_line)
                if processed_line: # Only add paragraph if there's content after inline formatting
                    parsed_html_blocks.append(f"<p>{processed_line}</p>")

            # Close any open list block at the end of the input
            if in_list_block:
                parsed_html_blocks.append('</ul>')

            final_html_content = "\n".join(parsed_html_blocks)
            logger.info("Successfully parsed Markdown content into HTML.")
            return {"html_content": final_html_content}

        except Exception as e:
            logger.error(f"An unexpected error occurred during Markdown parsing: {e}", exc_info=True)
            raise ValueError(f"Failed to parse Markdown content due to an internal error: {e}") from e

    def _apply_inline_formatting(self, text: str) -> str:
        """
        Applies basic inline Markdown formatting (bold, italic) to a given string.

        Args:
            text: The string segment to apply inline formatting to.

        Returns:
            The string with Markdown inline elements replaced by their HTML equivalents.
        """
        # Bold: **text** -> <strong>text</strong>
        # Uses a non-greedy match to handle multiple bold sections in a line.
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # Italic: *text* -> <em>text</em>
        # Uses a non-greedy match.
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        return text
