import logging
from typing import Any, Dict

# Assuming the vishustra_core.nodes.base_node module is discoverable
from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # This node has a hard dependency on the 'markdown' library.
    # If not installed, raise an ImportError at module load time.
    # In a real environment, this dependency would be specified in setup.py or requirements.txt.
    raise ImportError("The 'markdown' library is required. Please install it with 'pip install markdown'.")

# Set up logging for this module
logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node utilizes the `markdown` library to convert input Markdown
    strings into their corresponding HTML representations. It ensures
    robust handling of input data types and parsing failures.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting a Markdown string into an HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary containing contextual information. This node does
                     not currently utilize the context, but it is required by the
                     BaseNode interface.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an error occurs during the Markdown parsing process.
        """
        # Validate input data type
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Expected input data to be a string for Markdown parsing, "
                f"but received type: {type(data).__name__}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.debug(f"[{self.node_name}] Starting Markdown to HTML conversion.")

        try:
            # Perform the Markdown to HTML conversion using the 'markdown' library.
            # Additional extensions could be passed via context if the node were
            # configured to support them, e.g., markdown.markdown(data, extensions=context.get('markdown_extensions', []))
            html_output = markdown.markdown(data)
            logger.debug(f"[{self.node_name}] Successfully converted Markdown to HTML.")
            return html_output
        except Exception as e:
            # Catch any exceptions that might occur during the markdown conversion process
            error_msg = f"[{self.node_name}] Failed to parse Markdown: {e}"
            logger.error(error_msg, exc_info=True) # Log full traceback for debugging
            raise ValueError(f"Error parsing Markdown content: {e}") from e
