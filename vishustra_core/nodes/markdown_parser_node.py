import logging
from typing import Any, Dict, List

# Assuming 'markdown' is a standard dependency for rich text processing.
# In a real project, this would be listed in requirements.txt or pyproject.toml.
try:
    import markdown
except ImportError:
    # Provide a helpful error message if the dependency is missing
    raise ImportError(
        "The 'markdown' library is required for MarkdownParserNode. "
        "Please install it using 'pip install markdown'."
    )

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that converts Markdown formatted text into HTML.

    This node leverages the `markdown` library to perform the conversion,
    allowing for various Markdown extensions to be configured via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting Markdown text to HTML.

        The `data` input is expected to be a string containing Markdown content.
        The `context` can optionally contain a 'markdown_extensions' key,
        which should be a list of strings representing markdown extensions
        to be used (e.g., ['fenced_code', 'tables']).

        Args:
            data: The input Markdown string to be parsed.
            context: A dictionary containing operational context,
                     potentially including 'markdown_extensions'.

        Returns:
            The HTML string generated from the Markdown input.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during Markdown parsing.
        """
        logger.debug("MarkdownParserNode started processing.")

        if not isinstance(data, str):
            logger.error(
                "Invalid input data type for MarkdownParserNode. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"MarkdownParserNode expects input 'data' to be a string containing Markdown content, "
                f"but received type '{type(data).__name__}'."
            )

        markdown_extensions: List[str] = []
        if "markdown_extensions" in context:
            extensions_from_context = context["markdown_extensions"]
            if isinstance(extensions_from_context, list):
                markdown_extensions = [str(ext) for ext in extensions_from_context]
                logger.debug(
                    "Using markdown extensions from context: %s", markdown_extensions
                )
            else:
                logger.warning(
                    "Context key 'markdown_extensions' was provided but is not a list. "
                    "Expected a list of strings, got '%s'. Proceeding without extensions.",
                    type(extensions_from_context).__name__
                )

        try:
            logger.info("Converting Markdown to HTML using markdown library with extensions: %s", markdown_extensions)
            html_output = markdown.markdown(data, extensions=markdown_extensions)
            logger.debug("Successfully converted Markdown to HTML.")
            return html_output
        except Exception as e:
            logger.exception("An unexpected error occurred during Markdown parsing.")
            raise RuntimeError(f"Failed to parse Markdown content: {e}") from e