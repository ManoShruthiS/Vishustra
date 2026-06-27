import logging
from typing import Any, Dict

# Assuming vishustra_core is installed and its structure is accessible
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to summarize input text.

    This node takes a string as input data and simulates a text summarization
    process. For demonstration purposes, it currently produces a simplified
    summary by truncating the input. In a production scenario, this would
    integrate with an actual summarization model (e.g., via an external API
    or a locally loaded model).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by summarizing the text.

        Args:
            data: The input text to be summarized. Expected to be a string.
            context: A dictionary containing contextual information for processing.
                     Can be used to pass parameters like `summary_length` or
                     `summarization_strategy`.

        Returns:
            A string representing the summarized text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', got '%s'.",
                self.node_name,
                type(data).__name__
            )
            raise TypeError(f"TextSummarizerNode expects string input, got {type(data).__name__}.")

        if not data.strip():
            logger.warning("[%s] Received empty or whitespace-only string for summarization.", self.node_name)
            return ""

        logger.info("[%s] Initiating text summarization for input of length %d.", self.node_name, len(data))

        # --- Simulated Summarization Logic ---
        # In a real-world scenario, this would involve calling an LLM API
        # or a local summarization model.
        # For demonstration, we're taking a simple approach:
        # - Split text into sentences (naively by '.').
        # - Take a percentage of sentences, or a fixed number of sentences.

        # Configuration for summarization can come from context or defaults
        summary_ratio = context.get('summary_ratio', 0.25)  # e.g., keep 25% of sentences
        min_sentences = context.get('min_sentences', 2)
        max_sentences = context.get('max_sentences', 5)

        sentences = [s.strip() for s in data.split('.') if s.strip()]

        if not sentences:
            logger.debug("[%s] No valid sentences found in the input text after splitting.", self.node_name)
            return ""

        num_target_sentences = max(min_sentences, min(max_sentences, int(len(sentences) * summary_ratio)))
        
        # Ensure we don't try to take more sentences than available
        num_sentences_to_take = min(num_target_sentences, len(sentences))

        summarized_sentences = sentences[:num_sentences_to_take]
        summary = ". ".join(summarized_sentences)
        if summary and not summary.endswith('.'):
            summary += '.' # Ensure proper sentence termination

        logger.debug(
            "[%s] Summarized text from %d sentences to %d sentences. Result length: %d.",
            self.node_name,
            len(sentences),
            len(summarized_sentences),
            len(summary)
        )
        return summary

# Example usage (for testing, typically this would be orchestrated by Vishustra framework)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    summarizer = TextSummarizerNode()

    test_text_short = "This is a short sentence. It has two parts."
    test_text_medium = (
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used to display "
        "all letters of the alphabet. It's often used for typing practice and font demonstrations. "
        "Some people find it quite catchy. Others prefer different phrases for their tests."
    )
    test_text_long = (
        "In a quaint little village nestled between rolling hills and a winding river, "
        "lived a clockmaker renowned for his intricate timepieces. Each clock he crafted "
        "was a masterpiece, not just in its mechanical precision but also in its artistic design. "
        "The villagers would often gather around his workshop, mesmerized by the delicate "
        "gears and springs that came to life under his skillful hands. One day, a mysterious "
        "traveler arrived, carrying an ancient, broken pocket watch, said to belong to a king "
        "from a forgotten era. The clockmaker, intrigued by its history and complexity, "
        "accepted the challenge, dedicating weeks to its restoration, meticulously repairing "
        "each tiny component. When he finally presented the working watch, it chimed with a "
        "melody so beautiful, it was said to awaken long-lost memories in all who heard it. "
        "This event further cemented his legend, drawing apprentices from far and wide."
    )
    
    # Test cases
    print("--- Testing TextSummarizerNode ---")

    # Test 1: Basic summarization
    summary1 = summarizer.process(test_text_medium, {})
    print(f"\nOriginal (medium): {test_text_medium[:100]}...")
    print(f"Summary 1: {summary1}")

    # Test 2: Long text with default settings
    summary2 = summarizer.process(test_text_long, {})
    print(f"\nOriginal (long): {test_text_long[:100]}...")
    print(f"Summary 2: {summary2}")
    
    # Test 3: Long text with custom summary ratio
    summary3 = summarizer.process(test_text_long, {'summary_ratio': 0.5}) # Try to get 50%
    print(f"\nOriginal (long, custom ratio): {test_text_long[:100]}...")
    print(f"Summary 3 (50% ratio): {summary3}")

    # Test 4: Short text, should return most/all sentences within min/max
    summary4 = summarizer.process(test_text_short, {})
    print(f"\nOriginal (short): {test_text_short}")
    print(f"Summary 4: {summary4}")

    # Test 5: Empty string
    try:
        summary5 = summarizer.process("", {})
        print(f"\nSummary (empty string): '{summary5}'")
    except Exception as e:
        print(f"\nError processing empty string: {e}")

    # Test 6: Non-string input
    try:
        summarizer.process(12345, {})
    except TypeError as e:
        print(f"\nCaught expected error for non-string input: {e}")
    except Exception as e:
        print(f"\nCaught unexpected error for non-string input: {e}")

    # Test 7: Whitespace-only string
    try:
        summary7 = summarizer.process("    \n\t ", {})
        print(f"\nSummary (whitespace only): '{summary7}'")
    except Exception as e:
        print(f"\nError processing whitespace string: {e}")<ctrl63>