import logging
from typing import Any, Dict

# Assuming 'markdown' library is installed (pip install markdown)
import markdown

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the 'markdown' library to convert Markdown-formatted
    strings into their corresponding HTML representation, suitable for
    downstream rendering or further processing.

    Configuration can be passed via the 'context' dictionary, specifically
    for Markdown extensions.
    """

    _NODE_NAME = "MarkdownParserNode"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self._NODE_NAME

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Parses Markdown input data into an HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information or
                     configuration for the node.
                     It can include an 'extensions' key, which should be a list of
                     strings corresponding to Markdown extensions to use (e.g., ['fenced_code']).

        Returns:
            A string representing the HTML output of the parsed Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            Exception: For any errors encountered during markdown parsing by the
                       underlying 'markdown' library.
        """
        logger.debug(f"[{self.node_name}] Initiating markdown parsing process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"MarkdownParserNode expects input 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        # Extract markdown extensions from context if provided
        markdown_extensions = context.get("extensions", [])
        if not isinstance(markdown_extensions, list):
            logger.warning(
                f"[{self.node_name}] 'extensions' in context is not a list. "
                "Ignoring provided extensions and proceeding without them."
            )
            markdown_extensions = []
        elif markdown_extensions:
            logger.debug(
                f"[{self.node_name}] Applying markdown extensions: {markdown_extensions}"
            )

        try:
            # Convert markdown string to HTML using the 'markdown' library
            html_output = markdown.markdown(data, extensions=markdown_extensions)
            logger.debug(f"[{self.node_name}] Successfully parsed markdown data to HTML.")
            return html_output
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unhandled error occurred during markdown parsing."
            )
            # Re-raise the exception to allow upstream nodes or the orchestrator
            # to handle the processing failure.
            raise
