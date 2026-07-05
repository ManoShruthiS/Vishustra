import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists and contains BaseNode
from vishustra_core.nodes.base_node import BaseNode
import markdown # External library for Markdown parsing

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown text into HTML.

    This node utilizes the 'markdown' library to perform the conversion.
    It expects a string containing Markdown as input and produces an HTML string.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this Markdown parser node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data, converting Markdown text into its HTML equivalent.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing Markdown syntax.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the current processing flow. This node
                                     currently does not utilize the context for
                                     configuration, but it's available for
                                     future extensions (e.g., passing markdown extensions).

        Returns:
            Any: The resulting HTML string after parsing the Markdown input.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If any error occurs during the Markdown parsing process
                          using the 'markdown' library.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Received invalid input type. Expected a string for Markdown parsing, "
                "but got '%s'.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"MarkdownParserNode expects a string input, but received "
                f"{type(data).__name__}."
            )

        try:
            # Future enhancement: Markdown extensions could be passed via context, e.g.:
            # extensions = context.get("markdown_extensions", [])
            # html_output = markdown.markdown(data, extensions=extensions)
            
            html_output = markdown.markdown(data)
            logger.debug("[%s] Successfully parsed Markdown to HTML.", self.node_name)
            return html_output
        except Exception as e:
            logger.error(
                "[%s] An error occurred during Markdown parsing: %s",
                self.node_name,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to parse Markdown data in {self.node_name} due to: {e}"
            ) from e

