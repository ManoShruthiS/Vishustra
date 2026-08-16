import logging
from typing import Any, Dict

# External library for Markdown parsing.
# Users would typically install this via `pip install markdown`.
import markdown 

# Importing the BaseNode from Vishustra's core library.
from vishustra_core.nodes.base_node import BaseNode

# Set up a logger for this module to ensure all operational messages
# are captured by the Vishustra logging infrastructure.
logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node responsible for parsing Markdown-formatted
    text and converting it into its corresponding HTML representation.

    This node is crucial for workflows that involve processing text
    from various sources (e.g., user input, knowledge bases) and
    preparing it for display in web interfaces or for further HTML-based
    transformations.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a string containing Markdown text,
        and transforms it into an HTML string.

        Args:
            data (Any): The input data. This node specifically expects a
                        string containing Markdown content.
            context (Dict[str, Any]): A dictionary providing contextual
                                     information from the orchestration
                                     pipeline, which might include configuration
                                     or shared state. (Not used by this node
                                     for core logic, but passed along).

        Returns:
            Any: The HTML string resulting from the Markdown conversion.

        Raises:
            TypeError: If the input `data` is not a string, indicating an
                       invalid input type for Markdown parsing.
            ValueError: If an unexpected error occurs during the Markdown
                        parsing process, potentially due to issues with the
                        parsing library or malformed but syntactically valid
                        Markdown that causes internal errors.
        """
        logger.debug(f"[{self.node_name}] Initiating Markdown parsing process.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type for data. "
                f"Expected a string, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Handle cases where the input string might be empty or just whitespace.
        # markdown.markdown("") correctly returns "", but an explicit check
        # can provide clearer logging for such scenarios.
        if not data.strip():
            logger.warning(
                f"[{self.node_name}] Input data is an empty or whitespace-only "
                "string. Returning an empty HTML string."
            )
            return ""

        try:
            # Perform the Markdown to HTML conversion.
            # The 'markdown' library provides robust parsing capabilities.
            # Extensions can be added here if a specific flavour of Markdown
            # (e.g., GitHub Flavored Markdown) is required, e.g.,
            # markdown.markdown(data, extensions=['gfm']).
            parsed_html = markdown.markdown(data)
            logger.debug(f"[{self.node_name}] Successfully converted Markdown to HTML.")
            return parsed_html
        except Exception as e:
            # Catching a broad exception to ensure the node is resilient to
            # any unforeseen issues within the external markdown parsing library.
            error_msg = (
                f"[{self.node_name}] Failed to parse Markdown data due to an "
                f"unexpected error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e