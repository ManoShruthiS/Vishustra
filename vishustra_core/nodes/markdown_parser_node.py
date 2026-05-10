import logging
from typing import Any, Dict

try:
    import markdown
except ImportError:
    markdown = None

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node that transforms raw Markdown text into structured HTML.
    
    This node is designed to handle LLM-generated markdown, converting it into 
    a format suitable for web rendering or further downstream processing 
    within the Vishustra orchestration pipeline.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts input Markdown string into HTML using configurable extensions.

        Args:
            data: The raw Markdown string to be parsed.
            context: A dictionary containing execution metadata and configuration.
                     Supports 'markdown_extensions' key to customize the parser.

        Returns:
            A dictionary containing the parsed content, status, and metadata.

        Raises:
            TypeError: If the input data is not a string.
            RuntimeError: If the parsing library is missing or an internal error occurs.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Input validation failed. Expected str, got {type(data).__name__}.")
            raise TypeError(f"MarkdownParserNode requires string input, received {type(data).__name__}")

        if not data.strip():
            logger.info(f"[{self.node_name}] Received empty string. Skipping transformation.")
            return {
                "parsed_content": "",
                "metadata": {"empty_input": True},
                "status": "skipped"
            }

        if markdown is None:
            logger.error(f"[{self.node_name}] Critical dependency 'markdown' is not installed.")
            raise RuntimeError("The 'markdown' Python package is required for MarkdownParserNode.")

        try:
            # Extract configuration from context or use sensible defaults
            extensions = context.get("markdown_extensions", ["extra", "codehilite", "toc", "fenced_code"])
            output_format = context.get("markdown_output_format", "html5")

            logger.debug(f"[{self.node_name}] Parsing content with extensions: {extensions}")
            
            parsed_html = markdown.markdown(
                data, 
                extensions=extensions,
                output_format=output_format
            )

            return {
                "parsed_content": parsed_html,
                "metadata": {
                    "input_length": len(data),
                    "output_length": len(parsed_html),
                    "parser": "python-markdown"
                },
                "status": "success"
            }

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during markdown processing.")
            raise RuntimeError(f"Markdown parsing failed: {str(e)}") from e