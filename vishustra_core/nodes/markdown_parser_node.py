import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per Vishustra's project structure
from vishustra_core.nodes.base_node import BaseNode

try:
    import markdown
except ImportError:
    # In a production Vishustra environment, missing dependencies would typically
    # be caught during deployment or environment setup. This block serves as
    # an explicit reminder that 'markdown' is a required external library.
    # For a robust system, this could be handled by a dependency checker
    # at node registration time, or by ensuring all requirements are met.
    raise ImportError(
        "The 'markdown' library is required for MarkdownParserNode. "
        "Please install it using 'pip install markdown'."
    )


logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown formatted text into HTML.

    This node leverages the 'markdown' library to perform the conversion,
    ensuring robust and compliant HTML output from various Markdown inputs.
    It expects the input `data` to be a string containing Markdown syntax.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, parsing a Markdown string into an HTML string.

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context.

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input data is not a string.
            RuntimeError: If any unexpected error occurs during Markdown parsing.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' (Markdown text), but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Perform the Markdown to HTML conversion
            html_output = markdown.markdown(data)
            logger.info(f"[{self.node_name}] Successfully parsed Markdown data into HTML.")
            return html_output
        except Exception as e:
            # Catching a general Exception to cover any unforeseen issues
            # from the markdown library or underlying system calls.
            error_msg = f"[{self.node_name}] An error occurred during Markdown parsing: {e}"
            logger.exception(error_msg)  # Logs the full traceback
            raise RuntimeError(error_msg) from e
            # Re-raise with a specific Runtime error to allow upstream
            # orchestration to handle processing failures.
