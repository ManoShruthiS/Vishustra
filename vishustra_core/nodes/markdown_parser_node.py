import logging
from typing import Any, Dict

# External library for Markdown parsing.
# Ensure 'markdown' is installed: pip install markdown
try:
    import markdown
except ImportError:
    # Log a critical error and halt if the essential dependency is missing,
    # as this node cannot function without it.
    logging.getLogger(__name__).critical(
        "The 'markdown' library is not installed. Please install it using 'pip install markdown' "
        "to enable the MarkdownParserNode functionality."
    )
    raise

# Assume BaseNode is accessible relative to the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown formatted strings
    into their corresponding HTML representations.

    This node leverages the 'markdown' Python library for efficient and
    configurable conversion. It supports passing Markdown extensions and
    their configurations via the context dictionary, allowing for flexible
    parsing behaviors.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Parses the input data, expected to be a Markdown string, into an HTML string.

        The node can be configured via the `context` dictionary to use specific
        Markdown extensions and their configurations:
        - `context['markdown_extensions']`: An optional list of extension names
          (e.g., ['fenced_code', 'tables']).
        - `context['markdown_extension_configs']`: An optional dictionary of
          extension configurations, where keys are extension names and values
          are dictionaries of their respective settings.

        Args:
            data (Any): The input data. Expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                       potentially including 'markdown_extensions'
                                       and 'markdown_extension_configs' for configuration.

        Returns:
            str: The HTML string resulting from the Markdown parsing.

        Raises:
            TypeError: If the input `data` is not a string, as Markdown parsing
                       is only applicable to string content.
            ValueError: If an unexpected error occurs during the Markdown parsing
                        process, possibly due to malformed input or invalid
                        extension configurations.
        """
        if not isinstance(data, str):
            error_message = (
                f"{self.node_name} expects input data of type 'str' for parsing, "
                f"but received '{type(data).__name__}'. Aborting process."
            )
            logger.error(error_message, extra={"node_name": self.node_name, "received_type": type(data).__name__})
            raise TypeError(error_message)

        try:
            # Retrieve optional Markdown extensions and their configurations from context.
            # Default to empty lists/dictionaries if not provided.
            extensions = context.get('markdown_extensions', [])
            extension_configs = context.get('markdown_extension_configs', {})

            # Perform the Markdown to HTML conversion using the configured extensions.
            html_output = markdown.markdown(data, extensions=extensions, extension_configs=extension_configs)

            logger.info(
                f"Successfully parsed Markdown data into HTML using {self.node_name}.",
                extra={
                    "node_name": self.node_name,
                    "input_data_length": len(data),
                    "output_data_length": len(html_output),
                    "extensions_used": extensions
                }
            )
            return html_output

        except Exception as e:
            # Catch any unexpected errors that might occur during the markdown processing
            # (e.g., invalid extension names, issues within extension processing).
            error_message = (
                f"An unexpected error occurred during Markdown parsing in {self.node_name}: {e}. "
                "Review the input data or context configurations for 'markdown_extensions' "
                "and 'markdown_extension_configs'."
            )
            logger.error(error_message, exc_info=True, extra={"node_name": self.node_name, "error_type": type(e).__name__})
            # Re-raise as a ValueError, chaining the original exception for better debuggability.
            raise ValueError(error_message) from e