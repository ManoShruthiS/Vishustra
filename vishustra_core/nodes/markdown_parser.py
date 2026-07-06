import logging
from typing import Any, Dict
import markdown # Assuming the 'markdown' package is installed and available
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node that parses Markdown formatted text into HTML.

    This node takes a string containing Markdown and converts it into its
    corresponding HTML representation using the 'markdown' library. It can
    optionally accept a list of Markdown extensions via the 'context' dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "markdown_parser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data (expected to be a Markdown string) into HTML.

        Args:
            data: The input data, which must be a string containing Markdown text.
            context: A dictionary containing additional information or configuration.
                     It can include a key 'markdown_extensions' (list of str)
                     to specify extensions for the Markdown parser.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            ValueError: If the input data is not a string.
            RuntimeError: If an unexpected error occurs during the Markdown parsing process.
        """
        logger.debug(f"[{self.node_name}] Initiating Markdown parsing process.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Retrieve markdown extensions from context, defaulting to an empty list
        markdown_extensions = context.get("markdown_extensions", [])

        if not isinstance(markdown_extensions, list):
            logger.warning(
                f"[{self.node_name}] 'markdown_extensions' in context is not a list. "
                "Ignoring invalid value and proceeding without extensions."
            )
            markdown_extensions = []
        elif not all(isinstance(ext, str) for ext in markdown_extensions):
            logger.warning(
                f"[{self.node_name}] 'markdown_extensions' in context contains non-string elements. "
                "Filtering invalid values and proceeding with valid string extensions only."
            )
            markdown_extensions = [ext for ext in markdown_extensions if isinstance(ext, str)]


        try:
            logger.debug(
                f"[{self.node_name}] Parsing Markdown data using extensions: {markdown_extensions}"
            )
            # Perform the Markdown to HTML conversion
            html_output = markdown.markdown(data, extensions=markdown_extensions)

            logger.debug(f"[{self.node_name}] Markdown parsing successfully completed.")
            return html_output
        except Exception as e:
            # Catch any unexpected errors during parsing and re-raise as a RuntimeError
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}"
            )
            logger.exception(error_msg) # Logs the traceback automatically
            raise RuntimeError(error_msg) from e