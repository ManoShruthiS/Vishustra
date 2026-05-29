import logging
from typing import Any, Dict

# Assuming 'markdown' library is installed (e.g., pip install markdown)
import markdown

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that converts Markdown text into HTML.

    This node expects a string containing Markdown content as input and
    produces a string with the corresponding HTML representation.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts it to HTML.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context for configuration,
                     but it adheres to the standard node interface.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            ValueError: If the input 'data' is not a string.
            RuntimeError: If an unexpected error occurs during the markdown parsing process.
        """
        logger.debug(f"[{self.node_name}] Attempting to parse markdown content.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type. Expected a string "
                f"containing Markdown, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # The 'markdown' library provides a straightforward way to convert.
            # Additional configuration (e.g., extensions) could be passed via 'context'
            # if more advanced parsing capabilities were required.
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed markdown data to HTML.")
            return html_output
        except Exception as e:
            error_msg = (
                f"[{self.node_name}] An error occurred during markdown parsing: {e}"
            )
            logger.exception(error_msg)  # Log exception traceback
            raise RuntimeError(error_msg) from e