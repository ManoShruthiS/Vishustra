import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that converts Markdown formatted text
    into a cleaner, plain text representation by stripping common Markdown
    syntax.

    This node is useful for pre-processing text received from sources
    that might contain Markdown, preparing it for LLM consumption
    or further plain text processing stages.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown to plain text.

        Expects the input `data` to be a string containing Markdown.
        It strips various Markdown elements like headers, bold/italic markers,
        links (keeping only the text), lists, and code blocks.

        Args:
            data (Any): The input data, expected to be a string
                        containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual
                                      information for the processing.
                                      Not directly used by this node but
                                      available for future extensions.

        Returns:
            Any: A string containing the plain text representation of the
                 input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            Exception: For any unexpected errors during parsing.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected string, got {type(data)}.")
            raise TypeError(f"[{self.node_name}] Input 'data' must be a string, but received {type(data)}.")

        try:
            markdown_text: str = data

            # 1. Remove code blocks (multiline)
            # This regex matches blocks starting and ending with triple backticks.
            # It's a greedy match, so it will remove the entire block.
            markdown_text = re.sub(r'```.*?```', '', markdown_text, flags=re.DOTALL)
            # Remove inline code blocks (single backticks)
            markdown_text = re.sub(r'`[^`]+`', '', markdown_text)

            # 2. Convert headers: Remove leading '#' characters, keeping the text
            # and potentially adding a newline for separation.
            markdown_text = re.sub(r'^(#+\s*)(.*)', r'\2', markdown_text, flags=re.MULTILINE)

            # 3. Strip bold and italic markers: **, __, *, _
            markdown_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', markdown_text) # Bold (**) or (__)
            markdown_text = re.sub(r'(\*|_)(.*?)\1', r'\2', markdown_text)   # Italic (*) or (_)

            # 4. Convert links: [link text](URL) -> link text
            markdown_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)

            # 5. Remove images: ![]()
            markdown_text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', markdown_text)

            # 6. Remove list markers: -, *, +, numbers followed by a dot
            markdown_text = re.sub(r'^\s*[-*+]\s*', '', markdown_text, flags=re.MULTILINE)
            markdown_text = re.sub(r'^\s*\d+\.\s*', '', markdown_text, flags=re.MULTILINE)

            # 7. Remove blockquotes: >
            markdown_text = re.sub(r'^\s*>\s*', '', markdown_text, flags=re.MULTILINE)

            # 8. Remove horizontal rules: ---, ***, ___
            markdown_text = re.sub(r'^(\s*[-*_]\s*){3,}\s*$', '', markdown_text, flags=re.MULTILINE)

            # 9. Collapse multiple newlines into single newlines for cleaner output
            processed_text = re.sub(r'\n{2,}', '\n\n', markdown_text).strip()

            logger.debug(f"[{self.node_name}] Successfully parsed Markdown to plain text.")
            return processed_text
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during Markdown parsing.")
            raise Exception(f"[{self.node_name}] Failed to parse Markdown: {e}") from e

