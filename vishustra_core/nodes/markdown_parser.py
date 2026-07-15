import logging
from typing import Any, Dict

# Assuming 'vishustra_core.nodes.base_node' exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

# External dependency for Markdown parsing
try:
    import markdown
except ImportError:
    # This node explicitly requires the 'markdown' library to function.
    # It's better to fail early and clearly if a core dependency is missing.
    raise ImportError(
        "The 'markdown' library is required for MarkdownParserNode. "
        "Please install it using 'pip install markdown'."
    )

logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown content and converts it
    into HTML.

    This node leverages the 'markdown' library to perform the conversion,
    offering a robust way to transform textual data from Markdown syntax
    to standard HTML for further rendering or processing within the pipeline.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and converts it
        to an HTML string.

        Args:
            data: The input data, which is expected to be a string containing
                  Markdown content.
            context: A dictionary containing contextual information for the
                     node's processing. This node does not currently utilize
                     context for its core logic, but it's available for
                     future enhancements (e.g., passing markdown extensions).

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string, indicating an
                       incorrect data type for Markdown parsing.
            Exception: For any other unexpected errors that occur during the
                       Markdown to HTML conversion process, ensuring pipeline
                       failures are propagated.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input type: Expected string for Markdown parsing, "
                "but received type %s.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"{self.node_name} expects a string input for 'data', but "
                f"received {type(data).__name__}."
            )

        logger.debug("[%s] Initiating Markdown to HTML conversion.", self.node_name)

        try:
            # Perform the conversion from Markdown to HTML.
            # Additional configurations (e.g., extensions) could be passed
            # via the 'context' dictionary if needed for more flexible parsing.
            html_output = markdown.markdown(data)
            logger.info("[%s] Successfully converted Markdown to HTML.", self.node_name)
            return html_output
        except Exception as e:
            logger.exception(
                "[%s] An unexpected error occurred during Markdown to HTML conversion.",
                self.node_name,
            )
            # Re-raise the exception to ensure that issues are not silently
            # swallowed and the orchestration framework can handle them.
            raise
