import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

# Third-party library for Markdown parsing. This is a required dependency.
try:
    import markdown
except ImportError as e:
    # If the markdown library is not found, this node cannot function.
    # Log a critical error and re-raise to indicate a missing dependency.
    logging.getLogger(__name__).critical(
        "Failed to import 'markdown' library. "
        "The MarkdownParserNode requires this library to function. "
        "Please install it using 'pip install markdown'."
    )
    raise ImportError("Required 'markdown' library not found.") from e

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node is designed to transform input data, expected to be a string
    containing Markdown syntax, into its corresponding HTML representation.
    It leverages the `markdown` Python library for robust and reliable parsing.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, parsing Markdown text into HTML.

        Args:
            data: The input data. This node specifically expects a string
                  containing Markdown formatted text.
            context: A dictionary containing contextual information relevant to
                     the current orchestration run. This node does not explicitly
                     use the context but accepts it as per the BaseNode interface.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string, indicating an
                       incorrect data type for processing by this node.
            RuntimeError: If an unexpected error occurs during the Markdown
                          parsing operation, encapsulating the underlying exception.
        """
        logger.info("Node '%s': Initiating Markdown parsing process.", self.node_name)

        if not isinstance(data, str):
            logger.error(
                "Node '%s': Invalid input data type. Expected 'str', but received '%s'.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string. "
                f"Received type: '{type(data).__name__}'."
            )

        try:
            html_output = markdown.markdown(data)
            logger.info("Node '%s': Successfully converted Markdown to HTML.", self.node_name)
            return html_output
        except Exception as e:
            logger.exception( # Use logger.exception to automatically include traceback
                "Node '%s': An unhandled error occurred during Markdown to HTML conversion.",
                self.node_name
            )
            raise RuntimeError(
                f"Failed to parse Markdown using '{self.node_name}' due to an internal error."
            ) from e