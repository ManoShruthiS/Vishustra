from typing import Any, Dict
import logging

# We simulate the import path as per project context.
# In a real setup, this would be `from vishustra_core.nodes.base_node import BaseNode`
class BaseNode: # pragma: no cover - BaseNode is a placeholder here, actual import is from another file
    from abc import ABC, abstractmethod
    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass
# End of BaseNode placeholder
# --- Actual import for Vishustra would be:
# from vishustra_core.nodes.base_node import BaseNode


logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    This node takes a string as input and returns a summarized version
    based on a configurable word limit provided in the context.
    If no limit is provided, a default is used.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by summarizing it.

        Args:
            data (Any): The input data, expected to be a string of text.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Expected keys:
                                      - 'summary_word_limit' (int, optional): Maximum number
                                        of words for the summary. Defaults to 50.

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the 'summary_word_limit' in context is not a positive integer.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data)}. Expected a string.")
            raise TypeError(f"TextSummarizerNode requires string input, but received {type(data)}")

        summary_word_limit = context.get("summary_word_limit", 50)

        if not isinstance(summary_word_limit, int) or summary_word_limit <= 0:
            logger.warning(
                f"Invalid 'summary_word_limit' '{summary_word_limit}' provided in context for {self.node_name}. "
                "Falling back to default of 50 words. 'summary_word_limit' must be a positive integer."
            )
            summary_word_limit = 50

        words = data.split()
        if len(words) <= summary_word_limit:
            summary = data
            logger.debug(f"{self.node_name} processed text, no summarization needed (length within limit).")
        else:
            summary = " ".join(words[:summary_word_limit]) + "..."
            logger.info(f"{self.node_name} summarized text to {summary_word_limit} words.")

        return summary

# Example usage (for testing purposes, not part of the committed code):
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#
#     summarizer_node = TextSummarizerNode()
#
#     test_text_long = (
#         "Vishustra is a highly modular LLM orchestration framework written in Python. "
#         "It provides a robust and flexible way to chain together different "
#         "large language models and custom processing nodes to create complex "
#         "AI workflows. Developers can easily extend its capabilities by "
#         "implementing new nodes or integrating third-party services. "
#         "The framework emphasizes extensibility, maintainability, and scalability."
#         "This makes it suitable for a wide range of applications from chatbots to content generation."
#     )
#
#     test_text_short = "Short text example."
#     test_text_empty = ""
#
#     # Test case 1: Default summary length
#     summary1 = summarizer_node.process(test_text_long, {})
#     print(f"Summary (default 50 words):\n{summary1}\n")
#
#     # Test case 2: Custom summary length
#     summary2 = summarizer_node.process(test_text_long, {"summary_word_limit": 20})
#     print(f"Summary (20 words):\n{summary2}\n")
#
#     # Test case 3: Input shorter than limit
#     summary3 = summarizer_node.process(test_text_short, {"summary_word_limit": 20})
#     print(f"Summary (short text):\n{summary3}\n")
#
#     # Test case 4: Empty input
#     summary4 = summarizer_node.process(test_text_empty, {"summary_word_limit": 10})
#     print(f"Summary (empty text):\n'{summary4}'\n")
#
#     # Test case 5: Invalid summary_word_limit (non-int)
#     summary5 = summarizer_node.process(test_text_long, {"summary_word_limit": "abc"})
#     print(f"Summary (invalid limit type):\n{summary5}\n")
#
#     # Test case 6: Invalid summary_word_limit (negative)
#     summary6 = summarizer_node.process(test_text_long, {"summary_word_limit": -5})
#     print(f"Summary (negative limit):\n{summary6}\n")
#
#     # Test case 7: Non-string data
#     try:
#         summarizer_node.process(12345, {})
#     except TypeError as e:
#         print(f"Caught expected error: {e}")