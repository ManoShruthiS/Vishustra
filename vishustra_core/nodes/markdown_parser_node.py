import logging
from typing import Any, Dict

# External dependency for Markdown parsing
import markdown

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown formatted text
    into its corresponding HTML representation.

    This node is useful for converting user-provided or dynamically generated
    Markdown content into a displayable HTML format, often as an intermediary
    step before rendering or further HTML processing.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, parsing Markdown text into HTML.

        The node expects the 'data' parameter to be a string containing
        Markdown formatted content. It uses the `markdown` library to perform
        the conversion.

        Args:
            data: The input data, expected to be a string containing Markdown.
                  Non-string input will raise a TypeError.
            context: A dictionary containing contextual information for processing.
                     Currently not directly used by this node but available for
                     future extensions.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If an unexpected error occurs during the Markdown
                          parsing process.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name}: Received invalid data type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(
                error_msg,
                extra={
                    "node_name": self.node_name,
                    "input_type": type(data).__name__,
                    "data_sample": str(data)[:100]  # Log a small sample for context
                }
            )
            raise TypeError(error_msg)

        try:
            # Perform the Markdown to HTML conversion
            html_output = markdown.markdown(data)

            logger.info(
                f"{self.node_name}: Successfully parsed Markdown text to HTML. "
                f"Input length: {len(data)} characters, Output length: {len(html_output)} characters."
            )
            return html_output
        except Exception as e:
            # Catch any exceptions from the markdown library or other unexpected issues
            error_msg = (
                f"{self.node_name}: An unexpected error occurred during Markdown parsing: {e}"
            )
            logger.exception(
                error_msg,
                extra={
                    "node_name": self.node_name,
                    "error_type": type(e).__name__,
                    "error_detail": str(e),
                    "input_length": len(data)
                }
            )
            raise RuntimeError(error_msg) from e
