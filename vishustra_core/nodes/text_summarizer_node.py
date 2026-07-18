import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    This node takes a string as input data and returns a truncated version
    of the text, acting as a basic summarizer for demonstration purposes.
    In a real-world scenario, this would integrate with an LLM or a dedicated
    NLP library for sophisticated summarization.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by performing a simulated text summarization.

        Expects `data` to be a string. It truncates the input text to the
        first few sentences.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing operation. (Currently not used
                                       by this specific node's logic but available for future extensions).

        Returns:
            Any: A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting text summarization process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"TextSummarizerNode requires string input, but received {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only string for summarization.")
            return ""

        # --- Simulated Summarization Logic ---
        # This is a very basic simulation: take the first N sentences.
        # In a production system, this would involve an actual summarization model.
        sentences = re.split(r'(?<=[.!?])\s+', data)
        summary_length_sentences = context.get('summary_length_sentences', 3) # Configurable via context

        if len(sentences) <= summary_length_sentences:
            summary = data
            logger.info(f"[{self.node_name}] Text is short, returning original text as summary.")
        else:
            summary_sentences = sentences[:summary_length_sentences]
            summary = " ".join(summary_sentences)
            # Ensure summary ends with a period if not already present
            if not summary.endswith(('.', '!', '?')):
                summary += "..."
            logger.info(f"[{self.node_name}] Successfully summarized text to {summary_length_sentences} sentences.")

        logger.debug(f"[{self.node_name}] Finished summarization process.")
        return summary

# Example usage (for local testing, not part of the core library itself)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Simulate BaseNode for local testing as it's not present in this file's context
    class MockBaseNode(BaseNode):
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass # Not implemented for mock
        @property
        def node_name(self) -> str:
            return "MockNode"
    
    # Replace the actual import if running this standalone for testing purposes
    # from vishustra_core.nodes.base_node import BaseNode
    # This example assumes BaseNode is available or mocked
    
    summarizer_node = TextSummarizerNode()

    sample_text_short = "This is a short sentence. It has only two sentences. Too short to summarize much."
    sample_text_long = (
        "Vishustra is a highly modular LLM orchestration framework written in Python. "
        "It aims to provide a flexible and extensible architecture for building complex AI workflows. "
        "Each component within Vishustra is designed as a 'node' with a specific responsibility. "
        "These nodes can be chained together to form powerful data processing pipelines. "
        "The framework emphasizes ease of use, performance, and scalability across various LLM tasks. "
        "This is an extra sentence to make it even longer. One more for good measure!"
    )
    sample_text_empty = ""
    sample_text_whitespace = "   \n\t  "
    sample_data_invalid = 12345

    print(f"Node Name: {summarizer_node.node_name}")
    print("\n--- Processing Short Text ---")
    summary_short = summarizer_node.process(sample_text_short, {})
    print(f"Original: {sample_text_short}")
    print(f"Summary : {summary_short}")

    print("\n--- Processing Long Text (default 3 sentences) ---")
    summary_long = summarizer_node.process(sample_text_long, {})
    print(f"Original: {sample_text_long}")
    print(f"Summary : {summary_long}")
    
    print("\n--- Processing Long Text (custom 1 sentence) ---")
    summary_long_custom = summarizer_node.process(sample_text_long, {'summary_length_sentences': 1})
    print(f"Original: {sample_text_long}")
    print(f"Summary : {summary_long_custom}")

    print("\n--- Processing Empty Text ---")
    summary_empty = summarizer_node.process(sample_text_empty, {})
    print(f"Original: '{sample_text_empty}'")
    print(f"Summary : '{summary_empty}'")

    print("\n--- Processing Whitespace Text ---")
    summary_whitespace = summarizer_node.process(sample_text_whitespace, {})
    print(f"Original: '{sample_text_whitespace}'")
    print(f"Summary : '{summary_whitespace}'")

    print("\n--- Processing Invalid Data Type ---")
    try:
        summarizer_node.process(sample_data_invalid, {})
    except TypeError as e:
        print(f"Caught expected error: {e}")
