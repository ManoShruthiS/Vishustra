import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode
import markdown

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown text into other formats,
    primarily HTML, leveraging the `markdown` Python library.

    This node provides functionality to convert raw Markdown strings into structured
    HTML, supporting various Markdown extensions and their configurations passed
    through the processing context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data, which is expected to be a Markdown string,
        into an output format, defaulting to HTML.

        The `context` dictionary can be used to customize the parsing behavior,
        allowing for the inclusion of Markdown extensions and their specific
        configurations.

        Args:
            data: The Markdown string content to be parsed.
            context: A dictionary containing operational context and
                     optional configuration for the markdown parser.
                     Supported keys within `context`:
                     - 'extensions': Optional[List[str]] - A list of Markdown extension names
                                     to enable during parsing (e.g., ['fenced_code', 'tables']).
                     - 'extension_configs': Optional[Dict[str, Dict[str, Any]]] - A dictionary
                                            mapping extension names to their configuration dictionaries.
                     - 'output_format': Optional[str] - The desired output format (e.g., 'html').
                                        Note: The current implementation primarily generates HTML,
                                        and other formats requested will default to HTML output.

        Returns:
            The parsed content as a string, typically HTML.

        Raises:
            ValueError: If the input `data` is not a string, indicating an invalid input type.
            RuntimeError: If an unexpected error occurs during the markdown parsing process.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name}: Input data type mismatch. "
                f"Expected 'str' for Markdown content, but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Extract configuration from context, providing sensible defaults
            extensions: List[str] = context.get('extensions', [])
            extension_configs: Dict[str, Dict[str, Any]] = context.get('extension_configs', {})
            output_format: str = context.get('output_format', 'html').lower()

            if output_format != 'html':
                logger.warning(
                    f"{self.node_name}: Requested output format '{output_format}' is not "
                    f"fully supported by this node's direct implementation. "
                    f"Proceeding with default 'html' conversion."
                )

            # Perform the Markdown parsing
            parsed_content = markdown.markdown(
                data,
                extensions=extensions,
                extension_configs=extension_configs
            )

            logger.debug(
                f"{self.node_name}: Successfully parsed data. "
                f"Enabled extensions: {', '.join(extensions) if extensions else 'None'}."
            )
            return parsed_content

        except Exception as e:
            error_msg = f"{self.node_name}: Failed to process Markdown data. An unexpected error occurred: {e}"
            logger.exception(error_msg)  # Log the full traceback for debugging
            raise RuntimeError(error_msg) from e