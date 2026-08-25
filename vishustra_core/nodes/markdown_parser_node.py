
import logging
from typing import Any, Dict

# External dependency for Markdown parsing. Ensure it's installed: pip install markdown
import markdown

# Assuming BaseNode is located at vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra node designed to parse Markdown formatted text into HTML.

    This node expects a string containing valid Markdown content as input.
    It leverages the 'markdown' library to perform the conversion, outputting
    the corresponding HTML representation. Robust error handling is included
    to manage invalid input types and unexpected parsing issues.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name for this Markdown parser node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, interpreting it as Markdown, and converts it to HTML.

        This method is the core logic of the node. It validates the input type
        and utilizes the 'markdown' library to perform the conversion.
        Contextual information, such as a node identifier, is used for enhanced logging.

        Args:
            data (Any): The input data expected to be a string containing Markdown text.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the current processing pipeline run,
                                       e.g., node_id, flow_id, etc.

        Returns:
            Any: A string containing the HTML representation of the parsed Markdown.

        Raises:
            ValueError: If the input 'data' is not a string, indicating an
                        incorrect data type for Markdown parsing.
            RuntimeError: If an unexpected error occurs during the Markdown
                          parsing process, encapsulating the underlying exception.
        """
        # Retrieve a unique identifier for this node instance from the context for logging.
        # This helps in tracing logs within complex orchestration flows.
        node_id = context.get("node_id", f"Node:{self.node_name}")
        logger.info(f"[{node_id}] Starting Markdown parsing process.")

        # Validate input data type. Markdown parsing specifically requires string input.
        if not isinstance(data, str):
            error_message = (
                f"[{node_id}] Invalid input data type. Expected 'str' for Markdown "
                f"parsing, but received '{type(data).__name__}'. "
                f"Data received: {data!r}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        try:
            # Perform the Markdown to HTML conversion using the 'markdown' library.
            parsed_html = markdown.markdown(data)
            logger.info(f"[{node_id}] Successfully parsed Markdown content to HTML.")
            return parsed_html
        except Exception as e:
            # Catch any unexpected errors during the parsing process.
            # Log the full traceback for detailed debugging.
            error_message = (
                f"[{node_id}] An unexpected error occurred during Markdown parsing. "
                f"Input data length: {len(data)}. Error: {e}"
            )
            logger.exception(error_message)
            # Re-raise as a RuntimeError to signify a critical failure in the node's operation.
            raise RuntimeError(f"Markdown parsing failed for node '{node_id}': {e}") from e

