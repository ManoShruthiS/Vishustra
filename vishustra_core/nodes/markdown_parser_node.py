import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

# External library for robust Markdown parsing.
# This dependency would typically be listed in Vishustra's requirements.
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the 'markdown-it-py' library for robust and
    spec-compliant Markdown parsing, converting the input Markdown string
    into its corresponding HTML representation.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode and its internal markdown-it parser.
        The parser is configured with default settings.
        """
        self._md = MarkdownIt()
        logger.debug("MarkdownParserNode initialized.", extra={"node_name": self.node_name})

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
                  Non-string inputs will result in a ValueError.
            context: A dictionary of contextual information. This node does
                     not directly utilize the context but it is passed along
                     the orchestration chain.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            ValueError: If the input 'data' is not a string.
            Exception: For any unforeseen errors encountered during the
                       Markdown parsing process.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input type for MarkdownParserNode. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(
                error_msg,
                extra={"node_name": self.node_name, "input_type": type(data).__name__}
            )
            raise ValueError(error_msg)

        logger.info("Starting Markdown parsing.", extra={"node_name": self.node_name})
        try:
            html_output = self._md.render(data)
            logger.info(
                "Successfully parsed Markdown to HTML.",
                extra={"node_name": self.node_name, "output_length": len(html_output)}
            )
            return html_output
        except Exception as e:
            error_msg = f"Failed to parse Markdown due to an internal error: {e}"
            logger.error(
                error_msg,
                exc_info=True, # Logs the full traceback
                extra={"node_name": self.node_name, "error_type": type(e).__name__}
            )
            raise Exception(error_msg) from e
