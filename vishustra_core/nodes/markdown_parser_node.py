import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # Mark the 'markdown' library as unavailable if the import fails.
    # The node will then raise a RuntimeError during processing if this dependency is missing.
    markdown = None

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into HTML.

    This node leverages the external 'markdown' Python library to convert
    Markdown strings into their corresponding HTML representation, making
    it suitable for content transformation within LLM orchestration workflows.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts it to HTML.

        Args:
            data: The input data, which must be a string containing Markdown content.
            context: A dictionary containing additional runtime context or configuration
                     parameters for the node. Currently not utilized by this node,
                     but available for future extensions.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string, indicating an invalid
                       payload for markdown processing.
            RuntimeError: If the 'markdown' library is not installed, which is
                          a mandatory prerequisite for this node's functionality.
            Exception: Propagates any unexpected errors encountered during the
                       internal markdown parsing process, ensuring pipeline
                       failures are visible.
        """
        if markdown is None:
            # Log a critical error and raise if the required dependency is missing.
            error_msg = (
                f"[{self.node_name}] Required 'markdown' library is not installed. "
                "Please ensure it is installed using 'pip install markdown' "
                "to enable Markdown parsing."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        if not isinstance(data, str):
            # Log an error and raise if the input data type is incorrect.
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' for Markdown content but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            logger.debug(f"[{self.node_name}] Attempting to parse Markdown content.")
            # Perform the markdown to HTML conversion.
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed Markdown content into HTML.")
            return html_output
        except Exception as e:
            # Catch any exceptions during the markdown parsing itself and re-raise.
            error_msg = f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}"
            logger.exception(error_msg)  # Log with exception info for full traceback.
            raise # Re-raise the original exception to propagate the failure.