import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Attempt to import the markdown library. If not found, log a critical error
# and allow the class to be defined, but raise a RuntimeError during processing.
try:
    import markdown
except ImportError:
    markdown = None  # type: ignore
    _markdown_import_error = True
    logging.getLogger(__name__).critical(
        "The 'markdown' library is not installed. "
        "MarkdownParserNode will not function without it. "
        "Please install it using 'pip install markdown'."
    )
else:
    _markdown_import_error = False

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown content into HTML.

    This node expects a string containing Markdown as input data and
    transforms it into its corresponding HTML representation using the
    'markdown' Python library.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data, expecting a Markdown string, and returns
        its HTML representation.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. (Not used
                                       by this node, but required by BaseNode.)

        Returns:
            Any: A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If the 'markdown' library is not installed.
            Exception: Catches any unexpected errors during the parsing process.
        """
        if _markdown_import_error:
            logger.error("Attempted to process markdown, but 'markdown' library is not installed.")
            raise RuntimeError(
                "The 'markdown' library is required but not found. "
                "Please install it using 'pip install markdown'."
            )

        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for MarkdownParserNode. Expected string, "
                f"but received {type(data).__name__}. Data: {data!r}"
            )
            raise TypeError(
                f"MarkdownParserNode expects a string input, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.debug("Received empty or whitespace-only markdown string. Returning empty string.")
            return ""

        logger.debug(f"Parsing markdown content of length {len(data)}...")
        try:
            # The markdown library by default converts markdown to HTML
            html_output = markdown.markdown(data)
            logger.info("Successfully parsed markdown content to HTML.")
            return html_output
        except Exception as e:
            logger.exception(f"An unexpected error occurred during markdown parsing: {e}")
            raise # Re-raise the exception after logging for upstream handling
