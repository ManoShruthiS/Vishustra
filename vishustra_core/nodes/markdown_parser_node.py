import logging
from typing import Any, Dict

# This import assumes BaseNode is available at the specified path within the Vishustra project.
from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # If the 'markdown' library is not found, we set a flag and raise an error
    # during node initialization, as it's a core dependency for this node's function.
    markdown = None
    _markdown_missing = True
else:
    _markdown_missing = False

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node takes a string containing Markdown content and converts it
    into its corresponding HTML representation using the 'markdown' library.
    It's designed to be a fundamental building block for processing
    textual content in a pipeline where rich text formatting is needed.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode.

        This constructor performs a crucial check for the presence of the
        'markdown' library dependency. If the library is not found, the
        node cannot function, and a RuntimeError is raised to prevent
        instantiation of an inoperable node.
        """
        if _markdown_missing:
            error_message = (
                "MarkdownParserNode cannot be initialized: The 'markdown' library "
                "is not installed. Please install it using 'pip install markdown'."
            )
            logger.error(error_message)
            raise RuntimeError(error_message)
        logger.info("MarkdownParserNode initialized successfully. Ready to parse content.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to HTML.

        The node expects the input `data` to be a string containing Markdown-formatted
        text. It uses the `markdown` library to perform the conversion.
        The `context` dictionary is currently not utilized for parsing options but
        is included for future extensibility, allowing for dynamic configuration
        (e.g., markdown extensions, output format preferences).

        Args:
            data (Any): The input data, which must be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing context-specific
                                     information. Currently, this node does not
                                     interpret any keys from the context.

        Returns:
            Any: A string representing the HTML output generated from the input Markdown.

        Raises:
            ValueError: If the input `data` is not a string, indicating an invalid
                        payload for this parser node.
            RuntimeError: If the 'markdown' library was not successfully loaded during
                          node initialization. This should ideally be caught during `__init__`.
            RuntimeError: For any unexpected errors occurring during the markdown
                          parsing process.
        """
        logger.debug(f"MarkdownParserNode received data of type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for MarkdownParserNode. Expected a 'str', "
                f"but received '{type(data).__name__}'. Data: {data[:100]}..."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if _markdown_missing:
            # Defensive check; __init__ should prevent this state.
            error_msg = "MarkdownParserNode attempted to process content without the 'markdown' library being available."
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        logger.info("Beginning markdown parsing process to convert content to HTML.")
        try:
            # The 'markdown' library's default behavior is to convert Markdown to HTML.
            # In a future iteration, the 'context' could be used to pass
            # markdown extensions or other configuration parameters.
            # Example: html_output = markdown.markdown(data, extensions=context.get("markdown_extensions", []))
            html_output = markdown.markdown(data)
            logger.debug("Markdown content successfully parsed and converted to HTML.")
            return html_output
        except Exception as e:
            # Catching generic exceptions during the parsing process to provide robust error handling.
            error_msg = f"An unexpected error occurred during markdown parsing: {e}"
            logger.exception(error_msg) # Logs the full traceback for debugging.
            raise RuntimeError(error_msg) from e # Re-raise as a RuntimeError with original exception context.