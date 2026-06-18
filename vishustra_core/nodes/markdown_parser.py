
import logging
from typing import Any, Dict
import markdown

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node designed to parse Markdown text and convert it into HTML.

    This node leverages the 'markdown' library to perform the transformation,
    making it suitable for scenarios where formatted text needs to be rendered
    or processed further as HTML within the orchestration framework.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, parsing it as Markdown and converting it to HTML.

        Args:
            data: The input data, expected to be a string containing Markdown content.
            context: A dictionary containing contextual information for the node's
                     operation. This node does not currently utilize the context.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string, as Markdown parsing
                       requires string input.
            RuntimeError: If any unexpected error occurs during the Markdown
                          parsing process, encapsulating the underlying exception.
        """
        logger.debug(f"[{self.node_name}] Initiating Markdown parsing process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', received '{type(data).__name__}'. "
                "Markdown parsing requires string input."
            )
            raise TypeError(
                f"MarkdownParserNode expects 'data' to be a string for parsing, "
                f"but received type '{type(data).__name__}'."
            )

        try:
            # The 'markdown' library is assumed to be a pre-installed dependency.
            # Its presence is critical for this node's functionality.
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully converted Markdown data to HTML.")
            return html_output
        except Exception as e:
            # Catch-all for any potential issues during the markdown library's execution.
            logger.error(
                f"[{self.node_name}] An error occurred during Markdown parsing: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to parse Markdown data to HTML due to an internal error: {e}"
            ) from e

