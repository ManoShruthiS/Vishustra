import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that parses Markdown text and transforms it into
    a simplified HTML representation.
    
    This node expects a string containing Markdown as input and outputs
    a string with basic Markdown elements converted to HTML tags.
    It handles headers, bold, italics, links, and unordered lists.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to a simplified HTML string.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing execution context information.
                                      Currently not used by this node but available for future
                                      configuration or shared state.

        Returns:
            Any: A string representing the HTML output of the parsed Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If there's an unexpected error during markdown parsing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"MarkdownParserNode expects string input, but received {type(data).__name__}"
            )

        logger.info(f"[{self.node_name}] Starting Markdown parsing for input data.")

        try:
            html_output_lines: List[str] = []
            in_list: bool = False
            current_paragraph_lines: List[str] = []

            # Helper to flush any pending paragraph content
            def flush_paragraph():
                nonlocal current_paragraph_lines
                nonlocal html_output_lines
                if current_paragraph_lines:
                    # Join with space for multi-line paragraphs, then apply inline transforms
                    paragraph_text = " ".join(current_paragraph_lines)
                    html_output_lines.append(f"<p>{self._apply_inline_transforms(paragraph_text)}</p>")
                    current_paragraph_lines = []

            lines = data.strip().split('\n')

            for line in lines:
                stripped_line = line.strip()

                if not stripped_line:
                    # An empty line acts as a paragraph break or list terminator
                    if in_list:
                        html_output_lines.append("</ul>")
                        in_list = False
                    flush_paragraph()
                    continue

                # Headers (e.g., # Heading, ## Subheading)
                header_match = re.match(r'^(#+)\s*(.*)', stripped_line)
                if header_match:
                    if in_list: html_output_lines.append("</ul>"); in_list = False
                    flush_paragraph() # Close any pending paragraph before a new block element
                    level = len(header_match.group(1))
                    text = header_match.group(2).strip()
                    html_output_lines.append(f"<h{level}>{self._apply_inline_transforms(text)}</h{level}>")
                    continue

                # Unordered list items (e.g., - Item, * Item)
                list_item_match = re.match(r'^\s*[-*+]\s*(.*)', stripped_line)
                if list_item_match:
                    flush_paragraph() # Close any pending paragraph before a new block element
                    if not in_list:
                        html_output_lines.append("<ul>")
                        in_list = True
                    item_text = list_item_match.group(1).strip()
                    html_output_lines.append(f"<li>{self._apply_inline_transforms(item_text)}</li>")
                    continue
                
                # If we reach here, the line is regular text
                if in_list: # If a list was open but the current line is not a list item, close the list
                    html_output_lines.append("</ul>")
                    in_list = False
                
                current_paragraph_lines.append(stripped_line)

            # After the loop, flush any remaining open blocks (list or paragraph)
            if in_list:
                html_output_lines.append("</ul>")
            flush_paragraph()

            final_output = "\n".join(html_output_lines)
            logger.info(f"[{self.node_name}] Markdown parsing completed successfully.")
            return final_output

        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}",
                exc_info=True
            )
            raise ValueError(f"Failed to parse Markdown due to an internal error: {e}") from e

    def _apply_inline_transforms(self, text: str) -> str:
        """
        Applies inline Markdown transformations (bold, italic, links) to a given text.
        """
        # Bold: **text** -> <strong>text</strong>
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic: *text* -> <em>text</em>
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Basic links: [text](url) -> <a href="url">text</a>
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
        
        return text