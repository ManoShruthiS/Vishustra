import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path in the Vishustra project structure
# The BaseNode definition from the prompt is used for interface understanding.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse and convert Markdown formatted text
    into a simplified HTML-like structure or plain text, suitable for further
    processing or rendering.

    This node demonstrates basic Markdown syntax conversion for headers,
    bold, and italic text.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, treating it as Markdown text, and converts
        common Markdown elements into a simplified HTML-like string.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for the node.
                     Currently, not used for specific configuration but can be
                     leveraged for parsing options in the future.

        Returns:
            A string with basic Markdown syntax converted to HTML-like tags.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If parsing encounters an unexpected internal error.
        """
        logger.debug(f"[{self.node_name}] Starting Markdown parsing process.")
        logger.debug(f"[{self.node_name}] Context received: {context}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Data for MarkdownParserNode must be a string. "
                f"Got {type(data).__name__}"
            )

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only string for parsing.")
            return ""

        processed_text = data

        try:
            # Convert headers: ### Header -> <h3>Header</h3>, ## Header -> <h2>Header</h2>, # Header -> <h1>Header</h1>
            # Order matters: parse higher-level headers first to avoid partial matches
            processed_text = re.sub(r'^(###)\s*(.*)$', r'<h3>\2</h3>', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'^(##)\s*(.*)$', r'<h2>\2</h2>', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'^(#)\s*(.*)$', r'<h1>\2</h1>', processed_text, flags=re.MULTILINE)

            # Convert bold: **text** -> <strong>text</strong>
            processed_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_text)

            # Convert italic: *text* -> <em>text</em>
            processed_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_text)

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during Markdown parsing.")
            raise ValueError(f"Failed to parse Markdown due to an internal error: {e}") from e

        logger.debug(f"[{self.node_name}] Successfully completed Markdown parsing.")
        return processed_text

# Example of how to use the node (for testing purposes, not part of the core library)
if __name__ == "__main__":
    # Dummy BaseNode for local execution without the full vishustra_core
    # In a real Vishustra environment, this would be imported from the framework.
    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass

    # Re-declare MarkdownParserNode using the local BaseNode for standalone testing
    # This block would not be part of the actual file committed to Vishustra.
    class MarkdownParserNode(BaseNode):
        @property
        def node_name(self) -> str:
            return "MarkdownParser"
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            # Replicate the actual process method logic here for standalone test
            logger.debug(f"[{self.node_name}] (Local Test) Starting Markdown parsing process.")
            if not isinstance(data, str):
                raise TypeError(f"Data for MarkdownParserNode must be a string. Got {type(data).__name__}")
            
            processed_text = data
            processed_text = re.sub(r'^(###)\s*(.*)$', r'<h3>\2</h3>', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'^(##)\s*(.*)$', r'<h2>\2</h2>', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'^(#)\s*(.*)$', r'<h1>\2</h1>', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_text)
            processed_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_text)
            logger.debug(f"[{self.node_name}] (Local Test) Successfully completed Markdown parsing.")
            return processed_text

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    parser_node = MarkdownParserNode()

    test_markdown = """
# Welcome to Vishustra

This is a **paragraph** with *some italic text*.

## Subheading One

### Deeper Subheading

- Item 1
- Item 2

Another line of text.
    """

    print("\n--- Original Markdown ---")
    print(test_markdown)

    try:
        parsed_output = parser_node.process(test_markdown, {"metadata": "example"})
        print("\n--- Parsed Output ---")
        print(parsed_output)

        print("\n--- Testing with empty string ---")
        empty_output = parser_node.process("", {})
        print(f"Empty string output: '{empty_output}'")

        print("\n--- Testing with non-string data ---")
        try:
            parser_node.process(123, {})
        except TypeError as e:
            print(f"Caught expected error: {e}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during test execution: {e}")