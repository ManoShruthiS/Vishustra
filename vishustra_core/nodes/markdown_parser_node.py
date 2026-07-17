import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is discoverable in the project structure
from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # Log a critical error if the required external 'markdown' library is not found.
    # This node will not be functional without it.
    logging.critical("The 'markdown' library is required for MarkdownParserNode but is not installed. Please install it using 'pip install markdown'.")
    markdown = None # Set to None to ensure subsequent calls fail clearly

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node designed to parse Markdown formatted text into HTML.

    This node accepts a string input containing Markdown syntax and leverages
    the 'markdown' Python library to convert it into its corresponding HTML
    representation, making it suitable for display or further processing.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode.

        This constructor performs an initial check to ensure that the
        'markdown' library, a core dependency, is available. If not, it
        raises a RuntimeError to prevent instantiation of a non-functional node.
        """
        if markdown is None:
            # Re-raising specific error to ensure this is caught at instantiation if
            # the critical log message was missed.
            raise RuntimeError("MarkdownParserNode requires the 'markdown' library, which is not installed.")
        logger.debug("MarkdownParserNode initialized and 'markdown' library confirmed available.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to HTML.

        Args:
            data (Any): The input data expected to be a string containing Markdown.
                        If the input is not a string, a TypeError will be raised.
            context (Dict[str, Any]): A dictionary providing contextual information
                                      for the processing operation. This node currently
                                      does not utilize the context, but it's part of
                                      the BaseNode interface.

        Returns:
            Any: The resulting HTML string after parsing the Markdown input.

        Raises:
            TypeError: If the 'data' argument is not a string.
            ValueError: If an unexpected error occurs during the Markdown parsing
                        process by the underlying 'markdown' library.
        """
        logger.info(f"[{self.node_name}] Initiating Markdown parsing for incoming data.")

        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Invalid input data type. Expected a string "
                f"containing Markdown, but received {type(data).__name__}."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        try:
            # The markdown.markdown() function performs the conversion from
            # Markdown text to an HTML string.
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully converted Markdown to HTML.")
            return html_output
        except Exception as e:
            # Catching a broad exception to ensure robustness against unexpected
            # issues within the markdown library or its dependencies.
            error_message = (
                f"[{self.node_name}] An unexpected error occurred during Markdown "
                f"to HTML conversion: {e.__class__.__name__}: {e}"
            )
            logger.exception(error_message) # Logs traceback for debugging
            raise ValueError(error_message) from e