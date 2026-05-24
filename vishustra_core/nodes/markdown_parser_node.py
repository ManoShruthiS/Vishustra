import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # The 'markdown' library is a crucial dependency for this node.
    # If not found, we'll log a critical error and ensure process() raises an exception.
    markdown = None

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the 'markdown' Python library to convert Markdown
    formatted strings into their corresponding HTML representation.
    It supports configurable extensions via the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data (expected to be a Markdown string) into HTML.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Can include 'markdown_extensions' (list of strings)
                                      to enable specific Markdown extensions
                                      (e.g., `['fenced_code', 'tables']`).

        Returns:
            Any: The parsed HTML string.

        Raises:
            ValueError: If the input 'data' is not a string.
            RuntimeError: If the 'markdown' library is not installed or
                          if an unexpected error occurs during parsing.
        """
        logger.debug(f"[{self.node_name}] Starting Markdown parsing process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, received {type(data).__name__}."
            )
            raise ValueError(
                f"{self.node_name} requires string input, but received {type(data).__name__}."
            )

        if markdown is None:
            logger.critical(
                f"[{self.node_name}] The 'markdown' library is not installed. "
                "Cannot perform Markdown parsing. Please install it with 'pip install markdown'."
            )
            raise RuntimeError(
                f"[{self.node_name}] Required 'markdown' library not found. Please install it."
            )

        try:
            # Retrieve markdown extensions from context, default to an empty list
            extensions = context.get("markdown_extensions", [])
            if not isinstance(extensions, list):
                logger.warning(
                    f"[{self.node_name}] 'markdown_extensions' in context is not a list ({type(extensions).__name__}). "
                    "Ignoring and proceeding without extensions."
                )
                extensions = []

            parsed_html = markdown.markdown(data, extensions=extensions)
            logger.debug(f"[{self.node_name}] Successfully parsed Markdown data.")
            return parsed_html
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"[{self.node_name}] Failed to parse Markdown data: {e}") from e