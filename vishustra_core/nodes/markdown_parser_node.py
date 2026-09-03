import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse Markdown text into a simplified HTML string.
    This node implements basic Markdown-to-HTML conversion for common elements
    like headers, paragraphs, lists, bold, and italic text.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, which is expected to be a Markdown string,
        and converts it into a simplified HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary providing runtime context. This node does
                     not currently utilize the context, but it is required by
                     the BaseNode interface.

        Returns:
            A string representing the processed HTML.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If an unrecoverable issue occurs during the parsing process.
        """
        logger.info(f"[{self.node_name}] Starting Markdown parsing process for incoming data.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"but received {type(data).__name__}. Aborting parsing."
            )
            raise TypeError(f"Input data for {self.node_name} must be a string, got {type(data).__name__}.")

        try:
            markdown_text = data
            html_lines = []
            in_list = False
            line_num = 0

            # Normalize newlines for consistent splitting and process line by line
            processed_lines = markdown_text.replace('\r\n', '\n').split('\n')

            for line_num, line in enumerate(processed_lines):
                stripped_line = line.strip()

                if not stripped_line:
                    # If we were in a list, close it before processing empty lines
                    if in_list:
                        html_lines.append('</ul>')
                        in_list = False
                    continue # Skip empty lines

                # Headers (e.g., # H1, ## H2, ### H3, etc.)
                header_match = re.match(r'^(#+)\s+(.*)$', stripped_line)
                if header_match:
                    if in_list: # Close any open list before a new block element
                        html_lines.append('</ul>')
                        in_list = False
                    level = len(header_match.group(1))
                    content = header_match.group(2).strip()
                    html_lines.append(f"<h{level}>{content}</h{level}>")
                    continue

                # List items (unordered, e.g., - Item, * Item)
                list_item_match = re.match(r'^[*-]\s+(.*)$', stripped_line)
                if list_item_match:
                    if not in_list:
                        html_lines.append('<ul>')
                        in_list = True
                    content = list_item_match.group(1).strip()
                    # Apply inline formatting within list items
                    content = self._apply_inline_formatting(content)
                    html_lines.append(f"<li>{content}</li>")
                    continue

                # If not a header or list item, it's treated as a paragraph
                if in_list: # Close any open list before a new block element
                    html_lines.append('</ul>')
                    in_list = False

                # Apply inline formatting (bold, italic) to paragraph content
                formatted_line = self._apply_inline_formatting(stripped_line)
                html_lines.append(f"<p>{formatted_line}</p>")

            # Ensure any outstanding list is closed at the very end of the document
            if in_list:
                html_lines.append('</ul>')

            result_html = "\n".join(html_lines)
            
            logger.info(f"[{self.node_name}] Successfully parsed Markdown into simplified HTML.")
            return result_html
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing at line {line_num}: {e}",
                exc_info=True
            )
            raise ValueError(f"Failed to parse Markdown due to an internal error: {e}") from e

    def _apply_inline_formatting(self, text: str) -> str:
        """
        Applies basic inline formatting (bold, italic) to a given string using regex.
        This simplified implementation handles common patterns but does not attempt
        to resolve nested or complex Markdown edge cases.
        """
        # Bold: **text** -> <strong>text</strong>
        # Uses a non-greedy match (.*?) to correctly handle multiple bold instances on one line.
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # Italic: *text* -> <em>text</em>
        # Uses negative lookarounds to ensure it only matches single asterisks,
        # distinguishing them from double asterisks used for bold.
        text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
        
        return text