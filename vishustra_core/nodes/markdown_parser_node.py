import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown content.

    This node takes a string containing Markdown and transforms it into a
    basic HTML-like representation. It serves as a building block for
    content transformation pipelines, allowing subsequent nodes to work
    with a more structured or rendered format.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, interpreting it as Markdown and
        converting it to a simplified HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown syntax.
            context: A dictionary containing contextual information
                     relevant to the current orchestration run.

        Returns:
            A string representing the processed data, which is a basic
            HTML-like conversion of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an unexpected error occurs during the parsing simulation.
        """
        logger.debug(f"[{self.node_name}] Attempting to parse markdown from data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type for markdown parsing. "
                f"Expected `str`, but received `{type(data).__name__}`."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Simulate markdown parsing. For production, this would integrate
            # with a robust library like `markdown` or `commonmark`.
            # Here, we perform basic regex-based transformations.
            parsed_content = data

            # Convert `**bold**` to `<b>bold</b>`
            parsed_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', parsed_content)

            # Convert `*italic*` to `<i>italic</i>`
            # This regex is simplified and might overlap with bold for `***text***`
            parsed_content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', parsed_content)

            # Convert `# Header` to `<h1>Header</h1>`
            # Applies to the start of a line only.
            parsed_content = re.sub(r'^#\s*(.*)', r'<h1>\1</h1>', parsed_content, flags=re.MULTILINE)
            
            # Convert `## Subheader` to `<h2>Subheader</h2>`
            parsed_content = re.sub(r'^##\s*(.*)', r'<h2>\1</h2>', parsed_content, flags=re.MULTILINE)

            logger.info(f"[{self.node_name}] Successfully processed markdown content.")
            return parsed_content
        except Exception as e:
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during markdown parsing simulation: {e}"
            )
            logger.exception(error_msg) # Log the full traceback for debugging
            raise ValueError(error_msg) from e # Re-raise with original exception chained for context