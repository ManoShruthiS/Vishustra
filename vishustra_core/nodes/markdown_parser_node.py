import logging
from typing import Any, Dict

# Assume 'markdown' library is installed (e.g., via `pip install markdown`)
import markdown

# Vishustra framework base classes
from vishustra_core.nodes.base_node import BaseNode

# Configure logger for this module
logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown text
    and converting it into HTML format.

    This node utilizes the `markdown` Python library for robust conversion.
    It expects the input `data` to be a string containing valid Markdown syntax.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting a Markdown string into an HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary holding contextual information. This node does
                     not currently utilize the context, but it is available for
                     potential future configurations (e.g., `extensions` for markdown).

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            ValueError: If the input `data` is not a string.
            Exception: If an unexpected error occurs during the Markdown to HTML conversion.
        """
        logger.debug(
            "[%s] Attempting to process data of type: %s", self.node_name, type(data).__name__
        )

        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected a string, but received %s. "
                "Data must be a Markdown string for parsing.",
                self.node_name, type(data).__name__
            )
            raise ValueError(
                f"[{self.node_name}] Input data must be a string containing Markdown, "
                f"but got {type(data).__name__} instead."
            )

        try:
            # Use the markdown library to convert the input string to HTML.
            # Future enhancements could involve passing 'extensions' or 'output_format'
            # through the 'context' dictionary.
            html_output = markdown.markdown(data)
            logger.info("[%s] Successfully converted Markdown to HTML.", self.node_name)
            return html_output
        except Exception as e:
            logger.exception(
                "[%s] An unexpected error occurred during Markdown to HTML conversion.",
                self.node_name
            )
            # Re-raise the exception to propagate the error for upstream handling
            raise Exception(f"[{self.node_name}] Failed to parse Markdown: {e}") from e