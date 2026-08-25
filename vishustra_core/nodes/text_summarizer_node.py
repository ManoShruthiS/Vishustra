import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.
    This node takes an input text and condenses it based on configurable
    parameters provided in the context, aiming to extract a shorter,
    representative version of the original content.

    In a production LLM orchestration framework, this node would typically
    interface with an underlying LLM or a specialized summarization service.
    For this implementation, it provides a functional simulation of that
    behavior using simple word-count based truncation with some heuristics.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to produce a summarized text.

        The summarization logic is a simulation that truncates the text
        to a specified word count, influenced by a ratio, minimum, and
        maximum word limits. It attempts to end the summary at a natural
        sentence boundary.

        Args:
            data (Any): The input data expected to be a string containing
                        the text to be summarized.
            context (Dict[str, Any]): A dictionary containing parameters for
                                      the summarization process.
                                      Expected keys:
                                      - 'summary_ratio' (Optional[float]): The desired ratio
                                        of the summary's word count to the original text's
                                        word count (e.g., 0.3 for 30%). Defaults to 0.3.
                                        Must be between 0 and 1.
                                      - 'min_words' (Optional[int]): The minimum word count
                                        for the summary. Defaults to 10.
                                      - 'max_words' (Optional[int]): The maximum word count
                                        for the summary. Defaults to 100.

        Returns:
            str: The summarized text. If the input is too short, the original
                 text (stripped) is returned. If the input is empty or invalid,
                 an empty string or an error is raised respectively.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"Input data for '{self.node_name}' must be a string. Received: {type(data)}")
            raise ValueError(f"Invalid input data type for '{self.node_name}': Expected str, got {type(data)}.")

        stripped_data = data.strip()
        if not stripped_data:
            logger.info(f"Received empty or whitespace-only string for '{self.node_name}'. Returning empty string.")
            return ""

        text_words = stripped_data.split()
        original_word_count = len(text_words)

        # --- Configure summarization parameters from context with robust defaults ---
        summary_ratio = context.get('summary_ratio')
        if not isinstance(summary_ratio, (int, float)) or not (0 < summary_ratio < 1):
            summary_ratio = 0.3  # Default ratio
            logger.debug(f"Using default summary_ratio: {summary_ratio}. Context value was invalid or missing.")

        min_words = context.get('min_words')
        if not isinstance(min_words, int) or min_words < 0:
            min_words = 10  # Default minimum words
            logger.debug(f"Using default min_words: {min_words}. Context value was invalid or missing.")

        max_words = context.get('max_words')
        if not isinstance(max_words, int) or max_words < min_words:
            max_words = 100  # Default maximum words
            # Ensure max_words is never less than min_words
            if max_words < min_words:
                max_words = max(min_words, 100)
            logger.debug(f"Using default max_words: {max_words}. Context value was invalid or missing or less than min_words.")

        # --- Simulate summarization logic ---
        target_word_count = int(original_word_count * summary_ratio)
        # Apply min/max constraints
        target_word_count = max(min_words, min(target_word_count, max_words))

        if original_word_count <= target_word_count:
            logger.info(f"Original text for '{self.node_name}' is short ({original_word_count} words). Returning full text as summary.")
            return stripped_data # Return original if it's already short enough or shorter than target

        # Truncate the text to the target word count
        summarized_words = text_words[:target_word_count]
        summarized_text = " ".join(summarized_words)

        # Basic heuristic to make the summary end more naturally by finding
        # the last sentence-ending punctuation. This avoids abrupt truncations.
        # We search from the end of the (potentially truncated) string backwards.
        # This is a simple simulation, not a full NLP sentence tokenizer.
        last_sentence_end_match = re.search(r'[.!?](?=\s|$)', summarized_text[::-1])
        if last_sentence_end_match:
            # Calculate the original index in the non-reversed string
            # `start()` on reversed match gives position from end of original string
            original_idx = len(summarized_text) - last_sentence_end_match.start()
            summarized_text = summarized_text[:original_idx].strip()
        else:
            # If no clear sentence end, just add an ellipsis to indicate truncation
            summarized_text += "..."

        logger.info(f"Summarized text for '{self.node_name}' from {original_word_count} words to {len(summarized_words)} words.")
        return summarized_text.strip()