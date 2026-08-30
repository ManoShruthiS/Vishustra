
import re
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path at runtime within Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that parses Markdown text into a simplified HTML representation.
    This node simulates parsing by applying basic transformations to common Markdown elements,
    such as headers, lists, and inline formatting. It's designed to demonstrate
    how a full-featured Markdown parser would integrate into the Vishustra framework,
    without requiring external libraries or a complete Markdown specification implementation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts
        it into a simplified HTML string.

        This method performs a basic, regex-based simulation of Markdown parsing.
        For production-grade parsing, a dedicated library like 'markdown' or 'mistune'
        would typically be employed.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary of contextual information for processing.
                     Currently not utilized by this node but available for future
                     extensions (e.g., configuration options, shared state).

        Returns:
            A string representing the simplified HTML output. Returns an empty string
            if the input Markdown is empty or only contains whitespace.

        Raises:
            TypeError: If the input data is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting markdown parsing process for input data.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected string, got {type(data).__name__}.")
            raise TypeError(f"Input data for MarkdownParserNode must be a string, got {type(data).__name__}.")

        markdown_text = data.strip()
        if not markdown_text:
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only markdown string. Returning empty string.")
            return ""

        html_output_parts = []
        lines = markdown_text.split('\n')
        
        current_paragraph_lines = []
        in_list = False

        def flush_paragraph_and_list():
            nonlocal current_paragraph_lines, in_list
            if current_paragraph_lines:
                html_output_parts.append(f"<p>{' '.join(current_paragraph_lines)}</p>")
                current_paragraph_lines = []
            if in_list:
                html_output_parts.append('</ul>')
                in_list = False

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                flush_paragraph_and_list()
                continue

            # Header parsing (H1, H2, H3)
            header_match = re.match(r'^(#{1,3})\s+(.*)$', stripped_line)
            if header_match:
                flush_paragraph_and_list()
                level = len(header_match.group(1))
                content = header_match.group(2).strip()
                html_output_parts.append(f"<h{level}>{content}</h{level}>")
                continue

            # Unordered list item parsing
            list_item_match = re.match(r'^\s*[-*]\s+(.*)$', stripped_line)
            if list_item_match:
                if not in_list:
                    flush_paragraph_and_list() # Close any open paragraph
                    html_output_parts.append('<ul>')
                    in_list = True
                content = list_item_match.group(1).strip()
                html_output_parts.append(f"<li>{content}</li>")
                continue

            # If we're here, it's not a header or new list item. Close list if it was active.
            if in_list:
                html_output_parts.append('</ul>')
                in_list = False

            # Inline formatting for bold and italic within paragraphs/lines
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', stripped_line)
            processed_line = re.sub(r'\*(?!\*)(.*?)\*(?!\*)', r'<em>\1</em>', processed_line) # Matches single * not followed by another *

            current_paragraph_lines.append(processed_line)

        flush_paragraph_and_list() # Ensure any trailing content is processed

        parsed_html = "\n".join(html_output_parts)
        logger.info(f"[{self.node_name}] Successfully parsed markdown into simplified HTML.")
        return parsed_html
