import logging
from typing import Any, Dict

# External dependency for markdown parsing.
# Please ensure the 'markdown' library is installed (e.g., pip install markdown).
import markdown

# This is where BaseNode would be imported from in the Vishustra framework.
# For the purpose of this standalone file, we'll assume this path is correct.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the 'markdown' Python library to convert
    Markdown formatted strings into their corresponding HTML representation.
    It supports configurable Markdown extensions, allowing for flexible
    parsing based on context requirements.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode.
        No specific configuration is required at initialization, as
        parsing options (like extensions) are provided via the context
        in the process method, ensuring dynamic behavior.
        """
        super().__init__()
        logger.debug("MarkdownParserNode initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, parsing Markdown content into HTML.

        This method expects the 'data' parameter to be a string containing
        Markdown formatted text. It uses the `markdown` library to perform
        the conversion, optionally applying specified extensions and their
        configurations from the 'context' dictionary.

        Args:
            data (Any): The input data, which must be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing additional runtime information.
                Expected keys in context:
                - 'extensions' (optional, list[str]): A list of Markdown extension names
                  to enable (e.g., ['fenced_code', 'tables']). Defaults to an empty list.
                - 'extension_configs' (optional, dict): A dictionary mapping extension
                  names to their configuration dictionaries. Defaults to an empty dict.

        Returns:
            Any: A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string, as Markdown parsing
                       requires string input.
            RuntimeError: If an unexpected error occurs during the Markdown parsing
                          process. This typically indicates an issue with the
                          `markdown` library or its interaction with the input.
        """
        if not isinstance(data, str):
            error_msg = (
                f"MarkdownParserNode received invalid input type. "
                f"Expected 'str' for Markdown content, but got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Retrieve contextual parameters for markdown parsing
        extensions = context.get('extensions', [])
        extension_configs = context.get('extension_configs', {})

        logger.info(
            f"Attempting to parse markdown content with extensions: {extensions} "
            f"and extension configurations: {extension_configs}."
        )

        try:
            # The 'markdown' library is imported at the module level.
            # If it's not installed, an ImportError would occur during module loading.
            html_output = markdown.markdown(
                text=data,
                extensions=extensions,
                extension_configs=extension_configs
            )
            logger.debug("Successfully parsed markdown content to HTML.")
            return html_output
        except Exception as e:
            error_msg = f"Failed to parse markdown content due to an unexpected error: {e}"
            logger.exception(error_msg)  # Log with full traceback for debugging
            raise RuntimeError(error_msg) from e