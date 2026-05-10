import logging
from typing import Any, Dict, Optional

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that transforms raw Markdown strings into HTML or 
    structured dictionary representations.
    
    This node handles content sanitization and extension-based parsing 
    to facilitate downstream LLM context injection or UI rendering.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input Markdown data.
        
        Args:
            data: The raw markdown string to be parsed.
            context: A dictionary containing execution settings:
                - 'extensions': List of markdown extensions to use (default: ['extra', 'toc']).
                - 'output_format': Target format, though currently optimized for HTML.

        Returns:
            A dictionary containing the parsed content and metadata.

        Raises:
            ValueError: If the input data is not a string.
            RuntimeError: If parsing fails due to library or configuration issues.
        """
        if not isinstance(data, str):
            logger.error("MarkdownParserNode received non-string input. Type: %s", type(data))
            raise ValueError(f"MarkdownParserNode expected string input, got {type(data).__name__}")

        if not MARKDOWN_AVAILABLE:
            logger.error("The 'markdown' package is not installed in the current environment.")
            raise RuntimeError(
                "Markdown library is missing. Please install it using 'pip install markdown' "
                "to use the MarkdownParserNode."
            )

        logger.info("Initializing markdown parsing for payload of size: %d", len(data))

        try:
            # Extract configuration from context or use defaults
            extensions = context.get("markdown_extensions", ["extra", "toc", "codehilite"])
            extension_configs = context.get("markdown_extension_configs", {})

            # Execute transformation
            parsed_html = markdown.markdown(
                data, 
                extensions=extensions,
                extension_configs=extension_configs
            )

            result = {
                "status": "success",
                "output": parsed_html,
                "metadata": {
                    "input_length": len(data),
                    "extensions_used": extensions,
                    "node": self.node_name
                }
            }

            logger.debug("Successfully parsed markdown content.")
            return result

        except Exception as e:
            logger.exception("An unexpected error occurred during markdown transformation.")
            raise RuntimeError(f"Failed to process markdown: {str(e)}") from e

# End of file