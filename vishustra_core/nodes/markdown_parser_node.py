
import logging
from typing import Any, Dict

# Assuming 'markdown' is an installed dependency in the Vishustra environment
import markdown

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node that parses Markdown formatted text into HTML.

    This node expects a string containing Markdown as input and outputs the
    corresponding HTML string. It leverages the `markdown` library for parsing.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode.
        """
        super().__init__()
        # Specific logger for this node to allow granular logging configuration
        self._node_logger = logging.getLogger(f"{__name__}.{self.node_name}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to HTML.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for the process.
                     Currently not directly used by this node but available for
                     future extensions.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If an error occurs during the Markdown parsing process.
        """
        self._node_logger.debug(f"[{self.node_name}] Starting markdown parsing process.")

        if not isinstance(data, str):
            self._node_logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Aborting process."
            )
            raise TypeError(
                f"Input for '{self.node_name}' must be a string (Markdown text), "
                f"but received type '{type(data).__name__}'."
            )

        try:
            # Perform the Markdown to HTML conversion
            parsed_html = markdown.markdown(data)
            self._node_logger.info(f"[{self.node_name}] Successfully parsed markdown to HTML.")
            return parsed_html
        except Exception as e:
            self._node_logger.exception(
                f"[{self.node_name}] An unexpected error occurred during markdown parsing."
            )
            # Re-raise as a RuntimeError to signify a processing failure
            raise RuntimeError(f"Failed to parse markdown in '{self.node_name}': {e}") from e

