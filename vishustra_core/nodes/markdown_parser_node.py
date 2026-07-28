import logging
from typing import Any, Dict

# Assuming the 'markdown' library is installed.
# To install: pip install markdown
try:
    import markdown
except ImportError as e:
    # Re-raise with a more informative message if the core dependency is missing
    raise ImportError(
        "The 'markdown' library is required for MarkdownParserNode. "
        "Please install it using 'pip install markdown'."
    ) from e

# Import the base node from the specified project path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown text into HTML.

    This node leverages the 'markdown' library to provide robust conversion
    capabilities, supporting various Markdown features and extensions.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data, expected to be a Markdown string, into an HTML string.

        The `context` dictionary can be used to pass configuration,
        such as 'markdown_extensions' (a list of extension names to enable).

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary containing contextual information and optional
                     configuration for parsing.
                     Expected keys:
                     - 'markdown_extensions': (list[str], optional) A list of
                       Markdown extension names (e.g., ['fenced_code', 'tables']).

        Returns:
            A string containing the HTML representation of the parsed Markdown.

        Raises:
            TypeError: If the input data is not a string.
            Exception: For any errors encountered during the markdown parsing process.
        """
        if not isinstance(data, str):
            error_message = (
                f"Node '{self.node_name}' expects input 'data' to be a string, "
                f"but received type: {type(data).__name__}."
            )
            logger.error(
                error_message,
                extra={"node_name": self.node_name, "input_type": type(data).__name__}
            )
            raise TypeError(error_message)

        try:
            # Extract markdown extensions from context if provided
            markdown_extensions = context.get("markdown_extensions", [])
            if not isinstance(markdown_extensions, list):
                logger.warning(
                    f"Context key 'markdown_extensions' in node '{self.node_name}' "
                    "is not a list. Ignoring provided extensions.",
                    extra={"node_name": self.node_name}
                )
                markdown_extensions = []

            # Perform the markdown conversion
            parsed_html = markdown.markdown(data, extensions=markdown_extensions)

            logger.debug(
                f"Successfully parsed markdown data using node '{self.node_name}'. "
                f"Input length: {len(data)}, Output length: {len(parsed_html)}.",
                extra={"node_name": self.node_name, "input_len": len(data), "output_len": len(parsed_html)}
            )
            return parsed_html
        except Exception as e:
            error_message = (
                f"An unexpected error occurred during markdown parsing in "
                f"node '{self.node_name}': {e}"
            )
            logger.error(
                error_message,
                exc_info=True, # Log full traceback
                extra={"node_name": self.node_name, "error_type": type(e).__name__}
            )
            raise Exception(error_message) from e
