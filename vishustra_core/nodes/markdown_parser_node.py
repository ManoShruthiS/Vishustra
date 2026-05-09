import logging
from typing import Any, Dict, Optional
try:
    import markdown
except ImportError:
    markdown = None

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A node responsible for converting raw Markdown strings into HTML or 
    structured text to facilitate downstream consumption by LLMs or UI components.
    """

    def __init__(self, extensions: Optional[list] = None, output_format: str = "html5"):
        """
        Initializes the parser node with optional markdown extensions.
        
        Args:
            extensions: List of markdown extensions (e.g., ['extra', 'codehilite']).
            output_format: The desired output format, defaults to 'html5'.
        """
        self.extensions = extensions or ['extra', 'sane_lists', 'nl2br']
        self.output_format = output_format

        if markdown is None:
            logger.error("The 'markdown' package is not installed. MarkdownParserNode will fail.")

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Transforms markdown input into formatted HTML.

        Args:
            data: The raw markdown string to be processed.
            context: Execution context containing metadata or configurations.

        Returns:
            A dictionary containing the original source and the parsed output.

        Raises:
            ValueError: If the input data is not a string.
            ImportError: If the markdown library is missing at runtime.
        """
        if markdown is None:
            raise ImportError(
                "Required dependency 'markdown' is missing. "
                "Please install it via 'pip install markdown'."
            )

        if not isinstance(data, str):
            logger.error(f"Invalid data type received: {type(data)}. Expected string.")
            raise ValueError(f"MarkdownParserNode expects a string, but received {type(data)}.")

        try:
            logger.debug(f"Parsing markdown content of length: {len(data)}")
            
            # Convert markdown to HTML based on node configuration
            html_output = markdown.markdown(
                data,
                extensions=self.extensions,
                output_format=self.output_format
            )

            return {
                "source": data,
                "parsed_content": html_output,
                "format": self.output_format,
                "status": "success"
            }

        except Exception as e:
            logger.exception("Failed to parse markdown content.")
            return {
                "source": data,
                "error": str(e),
                "status": "error"
            }

    def __repr__(self) -> str:
        return f"<{self.node_name}(extensions={self.extensions})>"