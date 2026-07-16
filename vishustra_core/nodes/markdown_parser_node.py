import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into a structured dictionary
    representation. This node provides a simplified interpretation of common
    Markdown elements, suitable for further programmatic processing within the framework.

    It transforms a Markdown string into a list of dictionaries, where each dictionary
    represents a block-level element (e.g., heading, paragraph, list item) with
    its type, content, and relevant attributes.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode.
        """
        super().__init__()
        logger.debug(f"[{self.node_name}] Initializing MarkdownParserNode.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Union[str, int]]]:
        """
        Processes the input data (expected to be a Markdown string) and
        returns a structured list of dictionaries representing parsed blocks.

        Each dictionary in the output list describes a Markdown block element:
        - Headings: `{"type": "heading", "level": int, "content": str}`
        - List items: `{"type": "list_item", "marker": str, "content": str}`
        - Paragraphs: `{"type": "paragraph", "content": str}`
        - Error lines: `{"type": "error_line", "original_content": str, "error": str}`

        Example input:
        ```markdown
        # My Title

        This is a paragraph with **bold** text.

        - Item one
        - Item two
        ```

        Example output (simplified):
        ```python
        [
            {"type": "heading", "level": 1, "content": "My Title"},
            {"type": "paragraph", "content": "This is a paragraph with **bold** text."},
            {"type": "list_item", "marker": "-", "content": "Item one"},
            {"type": "list_item", "marker": "-", "content": "Item two"},
        ]
        ```

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.

        Returns:
            List[Dict[str, Union[str, int]]]: A list of dictionaries, where each
                                              dictionary represents a parsed Markdown block
                                              and its attributes.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for Markdown parsing."
            )

        parsed_blocks: List[Dict[str, Union[str, int]]] = []
        lines = data.split('\n')
        current_paragraph_lines: List[str] = []

        def _flush_paragraph():
            """Helper to add current_paragraph_lines as a paragraph block if non-empty."""
            if current_paragraph_lines:
                content = "\n".join(current_paragraph_lines).strip()
                if content:  # Only add if content is not just whitespace
                    parsed_blocks.append({"type": "paragraph", "content": content})
                current_paragraph_lines.clear()

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()

            try:
                # Attempt to match block-level elements
                # Heading detection (e.g., "# Heading", "## Subheading")
                heading_match = re.match(r"^(#+)\s(.*)", stripped_line)
                if heading_match:
                    _flush_paragraph()  # Flush any preceding paragraph
                    level = len(heading_match.group(1))
                    content = heading_match.group(2).strip()
                    parsed_blocks.append({"type": "heading", "level": level, "content": content})
                    continue

                # List item detection (e.g., "- Item", "* Item")
                list_match = re.match(r"^([-*+])\s(.*)", stripped_line)
                if list_match:
                    _flush_paragraph()  # Flush any preceding paragraph
                    content = list_match.group(2).strip()
                    parsed_blocks.append({"type": "list_item", "marker": list_match.group(1), "content": content})
                    continue

                # If an empty line, flush any accumulated paragraph.
                # This acts as a paragraph separator.
                if not stripped_line:
                    _flush_paragraph()
                    continue

                # If none of the above block types matched, consider it part of a paragraph.
                # We append the original line to preserve potential internal spacing/indentation
                # which might be relevant for paragraph formatting.
                current_paragraph_lines.append(line)

            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Error processing line {line_num+1}: '{line.strip()}'. Error: {e}",
                    exc_info=True
                )
                _flush_paragraph()  # Flush any partial paragraph before adding error
                parsed_blocks.append(
                    {"type": "error_line", "original_content": line.strip(), "error": str(e)}
                )

        # Flush any remaining paragraph content after processing all lines
        _flush_paragraph()

        logger.debug(f"[{self.node_name}] Successfully parsed input data into {len(parsed_blocks)} blocks.")
        return parsed_blocks