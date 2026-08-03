import logging
import re
from typing import Any, Dict

# Assuming BaseNode is correctly available at this path as per project structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.

    This node takes a string as input data and returns a summarized version.
    The summarization logic is a basic sentence extraction, configurable via
    the context dictionary.
    """

    def __init__(self):
        """Initializes the TextSummarizerNode."""
        super().__init__()
        logger.debug("TextSummarizerNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by performing a simulated text summarization.

        It expects the input `data` to be a string. The `context` dictionary
        can specify `summary_sentences` (int) to control the target number of
        sentences in the summary. If `data` is not a string, a TypeError is raised.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary for configuration.
                                     Expected keys:
                                     - 'summary_sentences' (int, optional): The desired number
                                       of sentences for the summary. Defaults to 3.

        Returns:
            str: The summarized text. Returns an empty string if input is empty or just whitespace.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        logger.info(f"Initiating process in {self.node_name} for input data type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = (
                f"Invalid input type for {self.node_name}. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        stripped_data = data.strip()
        if not stripped_data:
            logger.warning(f"Received empty or whitespace-only string for summarization in {self.node_name}.")
            return ""

        # Retrieve summary length from context or use a default
        summary_sentences = context.get("summary_sentences", 3)
        try:
            summary_sentences = int(summary_sentences)
            if summary_sentences <= 0:
                logger.warning(
                    f"Non-positive 'summary_sentences' value '{summary_sentences}' in context for {self.node_name}. "
                    "Defaulting to 3 sentences."
                )
                summary_sentences = 3
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid 'summary_sentences' value '{summary_sentences}' in context for {self.node_name}. "
                "Expected an integer. Defaulting to 3 sentences."
            )
            summary_sentences = 3

        logger.debug(f"Target summary length set to {summary_sentences} sentences.")

        # Simple sentence tokenization simulation (not a robust NLP tokenizer)
        # Using regex to split by common sentence terminators, keeping the terminator with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', stripped_data)
        
        # Filter out empty strings that might result from multiple terminators or leading/trailing spaces
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= summary_sentences:
            summarized_text = stripped_data
            logger.debug(f"Original text has {len(sentences)} sentences, which is less than or equal to "
                         f"the desired {summary_sentences}. Returning original text.")
        else:
            summarized_text = " ".join(sentences[:summary_sentences])
            logger.debug(f"Summarized text by taking the first {summary_sentences} sentences.")

        logger.info(f"Finished processing with {self.node_name}. Result length: {len(summarized_text)} characters.")
        return summarized_text

if __name__ == '__main__':
    # Example Usage (for local testing/demonstration)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock BaseNode if vishustra_core is not installed for local testing
    # In a real setup, this __main__ block wouldn't redefine BaseNode.
    from abc import ABC, abstractmethod
    class MockBaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
        @property
        @abstractmethod
        def node_name(self) -> str: pass
    
    # Temporarily patch BaseNode for local execution if it's the mock one
    if 'BaseNode' not in globals() or not issubclass(TextSummarizerNode, BaseNode):
        BaseNode = MockBaseNode # Type ignore needed if strict type checking

    summarizer = TextSummarizerNode()

    # Test Case 1: Standard summarization
    text1 = (
        "The quick brown fox jumps over the lazy dog. This is a very interesting "
        "sentence. We should definitely include it in the summary. Here is another "
        "sentence that might be important. Let's see how well it summarizes."
    )
    context1 = {"summary_sentences": 2}
    summary1 = summarizer.process(text1, context1)
    print(f"\nOriginal (2 sentences): {text1}")
    print(f"Summary (2 sentences): {summary1}")
    expected1 = "The quick brown fox jumps over the lazy dog. This is a very interesting sentence."
    assert summary1 == expected1
    print("Test 1 Passed.")

    # Test Case 2: Text shorter than desired summary length
    text2 = "Short text. Very short."
    context2 = {"summary_sentences": 5}
    summary2 = summarizer.process(text2, context2)
    print(f"\nOriginal (5 sentences): {text2}")
    print(f"Summary (5 sentences): {summary2}")
    assert summary2 == text2
    print("Test 2 Passed.")

    # Test Case 3: Empty string
    text3 = "   "
    context3 = {}
    summary3 = summarizer.process(text3, context3)
    print(f"\nOriginal (empty): '{text3}'")
    print(f"Summary (empty): '{summary3}'")
    assert summary3 == ""
    print("Test 3 Passed.")

    # Test Case 4: Invalid input type
    text4 = 123
    context4 = {}
    try:
        summarizer.process(text4, context4)
    except TypeError as e:
        print(f"\nCaught expected error: {e}")
        print("Test 4 Passed.")
    else:
        print("Test 4 Failed: TypeError was not raised.")

    # Test Case 5: Default summary length
    text5 = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    context5 = {} # No 'summary_sentences' means default to 3
    summary5 = summarizer.process(text5, context5)
    print(f"\nOriginal (default): {text5}")
    print(f"Summary (default): {summary5}")
    expected5 = "First sentence. Second sentence. Third sentence."
    assert summary5 == expected5
    print("Test 5 Passed.")

    # Test Case 6: Invalid summary_sentences in context
    text6 = "One. Two. Three. Four. Five."
    context6 = {"summary_sentences": "invalid"}
    summary6 = summarizer.process(text6, context6)
    print(f"\nOriginal (invalid context): {text6}")
    print(f"Summary (invalid context): {summary6}")
    expected6 = "One. Two. Three." # Should default to 3
    assert summary6 == expected6
    print("Test 6 Passed.")

    # Test Case 7: Zero summary_sentences in context
    text7 = "One. Two. Three. Four. Five."
    context7 = {"summary_sentences": 0}
    summary7 = summarizer.process(text7, context7)
    print(f"\nOriginal (zero context): {text7}")
    print(f"Summary (zero context): {summary7}")
    expected7 = "One. Two. Three." # Should default to 3
    assert summary7 == expected7
    print("Test 7 Passed.")

    print("\nAll tests complete.")