import logging
import re
from typing import Any, Dict

# Assuming BaseNode is correctly available at this path within the Vishustra project structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse basic Markdown syntax within a string
    and transform it into a simplified HTML-like string. This node focuses
    on common Markdown elements to produce a structured, readable output.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts it
        into a basic HTML-like string. The transformation includes:

        -   `# Header` -> `<h1>Header</h1>`
        -   `## Subheader` -> `<h2>Subheader</h2>`
        -   `- List Item` -> `<li>List Item</li>` (properly wrapped in `<ul>` tags)
        -   `**bold**` -> `<strong>bold</strong>`
        -   `*italic*` -> `<em>italic</em>`
        -   Other non-empty lines are wrapped in `<p>...</p>` tags.
        -   Empty lines are preserved for visual separation in the output.

        Args:
            data: The input data, which must be a string containing Markdown.
            context: A dictionary containing contextual information. This node
                     does not currently utilize specific context parameters, but
                     it is included as part of the `BaseNode` contract.

        Returns:
            A string representing the HTML-like parsed content. Each block element
            (headers, paragraphs, list items) will be on its own line for clarity.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "MarkdownParserNode received invalid input type. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError("MarkdownParserNode expects 'data' to be a string.")

        processed_lines = []
        lines = data.split('\n')
        in_list_block = False

        for line in lines:
            stripped_line = line.strip()

            # Check if we are exiting a list block
            if in_list_block and not stripped_line.startswith('- '):
                processed_lines.append('</ul>')
                in_list_block = False

            if not stripped_line:
                # Preserve blank lines for visual separation in the output
                processed_lines.append('')
                continue

            # Apply inline formatting first using non-greedy regular expressions
            # Bold: **text** -> <strong>text</strong>
            stripped_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', stripped_line)
            # Italic: *text* -> <em>text</em>
            stripped_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', stripped_line)

            # Apply block formatting based on line prefixes
            if stripped_line.startswith('# '):
                processed_lines.append(f"<h1>{stripped_line[2:].strip()}</h1>")
            elif stripped_line.startswith('## '):
                processed_lines.append(f"<h2>{stripped_line[3:].strip()}</h2>")
            elif stripped_line.startswith('- '):
                # Start a new unordered list if not already in one
                if not in_list_block:
                    processed_lines.append('<ul>')
                    in_list_block = True
                processed_lines.append(f"<li>{stripped_line[2:].strip()}</li>")
            else:
                processed_lines.append(f"<p>{stripped_line}</p>")
        
        # Ensure any open list block is closed at the end of the document
        if in_list_block:
            processed_lines.append('</ul>')

        return "\n".join(processed_lines)