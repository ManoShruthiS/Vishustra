import logging
from typing import Any, Dict

# Assuming 'vishustra_core.nodes.base_node' provides BaseNode
# For local development/testing, you might need to ensure this path is resolvable
# or temporarily create a dummy 'vishustra_core' structure.
from vishustra_core.nodes.base_node import BaseNode

# External dependency for Markdown parsing.
# This assumes 'markdown' library is installed (e.g., pip install markdown).
import markdown

# Configure a logger for this module.
logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown formatted text
    and converts it into HTML.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string,
        and returns its HTML representation.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing runtime context
                                       information (unused by this node currently).

        Returns:
            Any: A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If an error occurs during Markdown parsing.
        """
        logger.info(f"Node '{self.node_name}' initiated processing.")

        if not isinstance(data, str):
            error_message = (
                f"Node '{self.node_name}' received invalid data type. "
                f"Expected 'str' for Markdown content, but got '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        try:
            # Perform the Markdown to HTML conversion.
            # The 'context' could potentially be used to pass Markdown extensions,
            # but for a foundational node, a basic conversion is sufficient.
            html_output = markdown.markdown(data)
            logger.info(f"Node '{self.node_name}' successfully parsed Markdown content.")
            return html_output
        except Exception as e:
            # Catch any exceptions that might occur during the markdown conversion.
            error_message = f"Node '{self.node_name}' encountered an error while parsing Markdown: {e}"
            logger.exception(error_message)  # Log the exception with traceback
            raise RuntimeError(error_message) from e