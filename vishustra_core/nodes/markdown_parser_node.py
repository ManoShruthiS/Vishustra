
import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown text and
    transforming it into a basic HTML representation.

    This node provides a simulated conversion of common Markdown elements to HTML,
    making it suitable for ingestion into further processing steps that expect HTML.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts it
        into a dictionary containing the generated HTML string.

        This method performs a basic, line-by-line conversion of Markdown syntax
        to HTML. It handles headers, unordered lists, paragraphs, and common
        inline formatting such as bold, italic, links, and inline code.

        Args:
            data (Any): The input data, which is expected to be a string
                        containing Markdown content.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       or parameters for the processing operation.
                                       Currently, this node does not utilize context.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed HTML representation
                            under the 'html' key. Returns `{"html": ""}` for
                            empty or whitespace-only input.

        Raises:
            TypeError: If the input `data` is not a string, indicating an invalid
                       input type for Markdown parsing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input 'data' for '{self.node_name}' must be a string, "
                f"but received '{type(data).__name__}'."
            )

        if not data.strip():
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only Markdown content. "
                f"Returning empty HTML."
            )
            return {"html": ""}

        logger.info(f"[{self.node_name}] Starting Markdown parsing for the input data.")

        html_lines = []
        lines = data.split('\n')
        in_list = False

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                continue

            # Headers (e.g., # Heading, ## Subheading)
            if stripped_line.startswith('#'):
                match = re.match(r'(#+)\s*(.*)', stripped_line)
                if match:
                    level = len(match.group(1))
                    content = match.group(2).strip()
                    html_lines.append(f'<h{level}>{content}</h{level}>')
                    if in_list: # Headers break list context
                        html_lines.append('</ul>')
                        in_list = False
                    continue

            # Unordered lists (e.g., - Item, * Item)
            if stripped_line.startswith(('- ', '* ')):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                content = stripped_line[2:].strip()
                # Apply inline formatting to list item content
                content = self._apply_inline_formatting(content)
                html_lines.append(f'<li>{content}</li>')
                continue

            # If we reached here, it's not a header or list item.
            # Close any open list and treat it as a paragraph.
            if in_list:
                html_lines.append('</ul>')
                in_list = False

            # Paragraphs and inline formatting
            temp_line = self._apply_inline_formatting(stripped_line)
            html_lines.append(f'<p>{temp_line}</p>')

        # Close any list that might still be open at the end of the document
        if in_list:
            html_lines.append('</ul>')

        parsed_html = "\n".join(html_lines)
        logger.info(f"[{self.node_name}] Successfully parsed Markdown to HTML.")
        return {"html": parsed_html}

    def _apply_inline_formatting(self, text: str) -> str:
        """
        Applies basic inline Markdown formatting to a given text string.
        """
        # Bold: **text** -> <strong>text</strong> (order matters: bold before italic)
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # Italic: *text* -> <em>text</em>
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        # Links: [link text](url) -> <a href="url">link text</a>
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
        # Inline Code: `code` -> <code>code</code>
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        return text
