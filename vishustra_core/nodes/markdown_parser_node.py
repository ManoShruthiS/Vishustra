
import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that parses Markdown formatted text into a structured
    representation.
    
    This node simulates the conversion of Markdown input into a list of
    dictionaries, where each dictionary represents an identifiable element
    (e.g., header, paragraph, list item). It aims to provide a basic
    structured output that can be further processed or rendered.
    
    Input `data` is expected to be a string containing Markdown.
    Output will be a list of dictionaries, e.g.,
    `[{"type": "h1", "content": "My Title"}, {"type": "paragraph", "content": "Some text."}]`
    """
    
    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes the input data, converting Markdown text into a list of
        structured elements.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a parsed Markdown element with 'type'
                                  and 'content' keys.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error("MarkdownParserNode received non-string data. Type: %s", type(data))
            raise TypeError(
                f"Input data for MarkdownParserNode must be a string, got {type(data)}."
            )

        if not data.strip():
            logger.warning("MarkdownParserNode received empty or whitespace-only data, returning empty list.")
            return []

        parsed_elements: List[Dict[str, Any]] = []
        lines = data.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Simple header detection (e.g., # Header, ## Subheader)
            header_match = re.match(r'^(#+)\s*(.*)$', line)
            if header_match:
                level = len(header_match.group(1))
                content = header_match.group(2).strip()
                # Markdown spec typically supports H1 to H6
                if level > 6:
                    logger.warning("Header level %d detected, clamping to H6 for content: '%s'", level, content)
                    level = 6
                parsed_elements.append({"type": f"h{level}", "content": content})
                continue

            # Simple list item detection (e.g., - Item, * Item, 1. Item)
            list_item_match = re.match(r'^(-|\*|\d+\.)\s*(.*)$', line)
            if list_item_match:
                prefix = list_item_match.group(1)
                content = list_item_match.group(2).strip()
                item_type = "ordered_list_item" if re.match(r'\d+\.', prefix) else "unordered_list_item"
                parsed_elements.append({"type": item_type, "content": content})
                continue
            
            # Simple paragraph (default)
            # A real parser would handle bold, italic, links within paragraphs
            # For this simulation, we'll just treat the line as a paragraph
            parsed_elements.append({"type": "paragraph", "content": line})

        logger.info("Successfully parsed markdown data into %d elements.", len(parsed_elements))
        return parsed_elements

