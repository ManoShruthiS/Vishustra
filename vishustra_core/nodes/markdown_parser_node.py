import logging
from typing import Any, Dict

# Assuming the 'markdown' library is available in the project's environment.
# It provides robust functionality for parsing Markdown.
import markdown

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown formatted text
    into its corresponding HTML representation.

    This node leverages the common 'markdown' library to ensure accurate and
    feature-rich conversion, supporting various Markdown extensions through
    the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts it to HTML.

        The `context` dictionary can be used to pass configuration specific to
        the markdown parser, such as a list of extensions.

        Args:
            data (Any): The input data. This node expects a string containing
                        Markdown text.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       or configuration for the processing step.
                                       Expected keys might include:
                                       - 'markdown_extensions' (list[str]): A list of
                                         extension names to use with the markdown parser
                                         (e.g., ['fenced_code', 'tables']).

        Returns:
            Any: The processed output, which will be an HTML string representing
                 the parsed Markdown.

        Raises:
            TypeError: If the input 'data' is not a string, as Markdown parsing
                       is inherently string-based.
            Exception: Captures and re-raises any underlying exceptions that occur
                       during the Markdown parsing process (e.g., issues with extensions).
        """
        if not isinstance(data, str):
            error_msg = (
                f"MarkdownParserNode received unexpected input type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_msg, extra={"node_name": self.node_name, "input_type": type(data).__name__})
            raise TypeError(error_msg)

        try:
            # Extract markdown extensions from context, if provided
            extensions = context.get('markdown_extensions', [])
            if extensions:
                logger.debug(
                    f"MarkdownParserNode using extensions: {extensions}",
                    extra={"node_name": self.node_name, "extensions": extensions}
                )

            # Perform the Markdown to HTML conversion
            html_output = markdown.markdown(data, extensions=extensions)

            logger.debug(
                "Successfully parsed Markdown data into HTML.",
                extra={"node_name": self.node_name, "input_length": len(data), "output_length": len(html_output)}
            )
            return html_output
        except Exception as e:
            error_msg = (
                f"An error occurred during Markdown parsing in MarkdownParserNode: {e}"
            )
            logger.error(error_msg, exc_info=True, extra={"node_name": self.node_name})
            raise
