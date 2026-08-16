import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # Log an error if the 'markdown' library is not installed and provide guidance.
    # This ensures the node gracefully handles missing dependencies.
    logging.getLogger(__name__).error(
        "The 'markdown' library is not installed. Please install it using 'pip install markdown' "
        "to enable the MarkdownParserNode functionality."
    )
    # Define a dummy markdown function to prevent NameError, but raise an error if invoked.
    def _markdown_unavailable_func(*args, **kwargs):
        raise ImportError("The 'markdown' library is not installed.")
    # Create a mock module object with the unavailable function
    class _MockMarkdownModule:
        markdown = _markdown_unavailable_func
    markdown = _MockMarkdownModule()


logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown formatted text
    into its corresponding HTML representation.

    This node leverages the 'markdown' Python library. It expects string
    input containing Markdown syntax and produces an HTML string.
    Configuration for the underlying markdown parser, such as enabling
    specific extensions, can be passed through the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, which is expected to be a Markdown string,
        and converts it into an HTML string.

        The `context` dictionary can optionally contain a key
        'markdown_parser_config'. Its value should be a dictionary of
        keyword arguments to be passed directly to the `markdown.markdown()`
        function (e.g., `{'extensions': ['extra', 'nl2br']}`).

        Args:
            data: The input data, anticipated to be a string containing Markdown.
            context: A dictionary providing contextual information and
                     configuration for the node, potentially including
                     `markdown_parser_config`.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ImportError: If the 'markdown' library is not installed, preventing
                         parsing functionality.
            RuntimeError: If an unexpected error occurs during the Markdown
                          parsing process itself.
        """
        if not isinstance(data, str):
            logger.error(
                "MarkdownParserNode received non-string data. Expected a string, got type '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"MarkdownParserNode expects string input for Markdown parsing, "
                f"but received type {type(data).__name__}."
            )

        parser_config = context.get("markdown_parser_config", {})
        if not isinstance(parser_config, dict):
            logger.warning(
                "Invalid 'markdown_parser_config' found in context for MarkdownParserNode. "
                "Expected a dictionary, but received type '%s'. Ignoring configuration.",
                type(parser_config).__name__
            )
            parser_config = {}

        try:
            logger.debug(
                "Attempting to parse Markdown data using config: %s", parser_config
            )
            html_output = markdown.markdown(data, **parser_config)
            logger.info("Successfully parsed Markdown data to HTML.")
            return html_output
        except ImportError as ie:
            # Re-raise ImportError if it originated from the dummy function due to missing library
            logger.exception("Markdown parsing failed due to missing 'markdown' library.")
            raise ie
        except Exception as e:
            # Catch any other unexpected errors during the markdown parsing process.
            logger.error(
                "An unexpected error occurred during Markdown parsing: %s", e, exc_info=True
            )
            raise RuntimeError(f"Failed to parse Markdown data: {e}") from e