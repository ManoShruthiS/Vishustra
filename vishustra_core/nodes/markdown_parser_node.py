import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that converts Markdown formatted text into
    a simplified HTML representation.

    This node expects the input `data` to be a string containing Markdown.
    It performs a basic transformation, handling common elements like headers,
    bold, and italic text, and paragraphs.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to simplified HTML.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary for shared context, not directly
                                      used by this node for parsing logic but
                                      available for future extensions.

        Returns:
            Any: A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected str, got {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        html_lines = []
        lines = data.split('\n')
        
        in_paragraph = False

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                if in_paragraph:
                    html_lines.append("</p>")
                    in_paragraph = False
                continue

            # Headers
            if stripped_line.startswith('# '):
                if in_paragraph:
                    html_lines.append("</p>")
                    in_paragraph = False
                html_lines.append(f"<h1>{stripped_line[2:].strip()}</h1>")
                continue
            elif stripped_line.startswith('## '):
                if in_paragraph:
                    html_lines.append("</p>")
                    in_paragraph = False
                html_lines.append(f"<h2>{stripped_line[3:].strip()}</h2>")
                continue
            elif stripped_line.startswith('### '):
                if in_paragraph:
                    html_lines.append("</p>")
                    in_paragraph = False
                html_lines.append(f"<h3>{stripped_line[4:].strip()}</h3>")
                continue
            # ... can extend for H4-H6

            # Paragraphs and inline formatting
            if not in_paragraph:
                html_lines.append("<p>")
                in_paragraph = True

            processed_line = stripped_line
            # Basic bold: **text** -> <strong>text</strong>
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_line)
            # Basic italic: *text* -> <em>text</em> (only if not already bold markers)
            processed_line = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', processed_line)
            
            html_lines.append(processed_line)

        if in_paragraph:
            html_lines.append("</p>")

        result_html = "\n".join(html_lines)
        logger.debug(f"[{self.node_name}] Successfully processed data to HTML.")
        return result_html
