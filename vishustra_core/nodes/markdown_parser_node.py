import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

# A robust markdown parsing library for converting markdown to HTML
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages a robust markdown parser to convert input markdown strings
    into their corresponding HTML representations, making it suitable for content
    rendering pipelines. It's designed to be stateless concerning the markdown
    parsing process, treating each `process` call independently.
    """

    def __init__(self):
        """
        Initializes the MarkdownParserNode and its internal markdown parser instance.
        """
        # We instantiate the markdown parser here to reuse it across process calls,
        # avoiding overhead of re-initialization.
        self._parser = MarkdownIt()
        logger.debug("MarkdownParserNode initialized with markdown-it-py parser.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data (expected to be a Markdown string) into HTML.

        Args:
            data: The input data, expected to be a string containing Markdown content.
            context: A dictionary containing contextual information for the node's
                     operation. While not directly used by this specific node, it
                     is part of the standard `BaseNode` API contract.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string, as this node
                       specifically requires string input for markdown parsing.
            RuntimeError: If an unexpected error occurs during the markdown
                          parsing process itself, indicating a failure to convert.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, but received {type(data).__name__}."
            )
            raise TypeError(
                f"{self.node_name} expects input data of type 'str', "
                f"but received {type(data).__name__}."
            )

        markdown_content: str = data
        parsed_html: str = ""
        try:
            parsed_html = self._parser.render(markdown_content)
            logger.info(f"[{self.node_name}] Successfully parsed markdown content.")
        except Exception as e:
            # Catching a broad exception to ensure robustness against unexpected
            # issues within the external markdown parsing library.
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during "
                f"markdown parsing: {e}"
            )
            raise RuntimeError(f"Failed to parse markdown content in {self.node_name}.") from e

        return parsed_html
