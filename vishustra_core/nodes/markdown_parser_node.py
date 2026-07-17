
import re
import logging
from typing import Any, Dict, List

# Assuming BaseNode is available at this path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that parses Markdown input and transforms it into a
    cleaned, more structured plain-text representation suitable for further
    processing, such as feeding into an LLM or for content extraction.

    This node provides a basic simulation of Markdown parsing, converting
    common Markdown elements into a simplified plain-text format.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and returns
        a cleaned plain-text version.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for the node.
                     (Currently not used by this node but available for future extensions).

        Returns:
            A string representing the plain-text version of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an unexpected error occurs during parsing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type for 'data'. Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string. Got {type(data).__name__}."
            )

        markdown_text: str = data
        cleaned_lines: List[str] = []

        try:
            # Split into lines to process line-by-line for some elements
            lines = markdown_text.split('\n')

            for line in lines:
                processed_line = line.strip()

                # Headers (e.g., # Header, ## Subheader)
                processed_line = re.sub(r'^(#+)\s*(.*)$', lambda m: f"{'#' * len(m.group(1))} {m.group(2).upper()}", processed_line)
                
                # Blockquotes (e.g., > Quote)
                processed_line = re.sub(r'^\s*>\s*(.*)$', r'[QUOTE] \1', processed_line)

                # Unordered lists (e.g., - Item, * Item)
                processed_line = re.sub(r'^(\s*)[-\*+]\s+(.*)$', r'\1* \2', processed_line)

                # Ordered lists (e.g., 1. Item)
                processed_line = re.sub(r'^(\s*)\d+\.\s+(.*)$', r'\11. \2', processed_line)

                # Code blocks (simple stripping of fences)
                # This simplistic approach just removes the fences and keeps content.
                # A more sophisticated parser might preserve indentation or language hints.
                if processed_line.startswith('```'):
                    processed_line = "" # Remove fence lines completely

                # Bold and Italic (e.g., **bold**, *italic*, __bold__, _italic_)
                # Prioritize bold, then italic
                processed_line = re.sub(r'(\*\*|__)(.*?)\1', r'**\2**', processed_line)
                processed_line = re.sub(r'(\*|_)(.*?)\1', r'*\2*', processed_line)

                # Links (e.g., [text](url)) - keep only the text
                processed_line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', processed_line)

                # Images (e.g., ![alt text](url)) - keep only the alt text
                processed_line = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', processed_line)

                # Inline code (e.g., `code`)
                processed_line = re.sub(r'`(.*?)`', r'\\`\1\\`', processed_line) # Escape backticks for distinction

                # Horizontal Rules (e.g., ---)
                if re.match(r'^-{3,}|^\*{3,}|^_{3,}$', processed_line):
                    processed_line = "---" # Normalize HR

                # Remove extra spaces but preserve single spaces between words
                processed_line = re.sub(r'\s{2,}', ' ', processed_line).strip()

                if processed_line: # Only add non-empty lines
                    cleaned_lines.append(processed_line)

            # Rejoin lines, ensure paragraphs are separated by at least one newline
            parsed_text = "\n".join(cleaned_lines)
            
            logger.debug(f"[{self.node_name}] Successfully parsed markdown data.")
            return parsed_text
        except Exception as e:
            logger.error(f"[{self.node_name}] An error occurred during markdown parsing: {e}", exc_info=True)
            raise ValueError(f"[{self.node_name}] Failed to parse markdown data due to an internal error.") from e

