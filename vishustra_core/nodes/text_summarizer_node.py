import logging
from typing import Any, Dict

# Assuming BaseNode is in this path relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates text summarization.

    This node is designed to take a string as input and return a summarized version.
    For the current implementation, it simulates summarization by truncating the
    text based on a configurable maximum word count or a ratio of the original text
    length, appending an ellipsis to indicate truncation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to generate a simulated summary.

        This method expects a string as input `data`. It truncates the text
        to simulate summarization. The length of the summary can be controlled
        via the `context` dictionary.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       and configuration parameters for summarization.
                                       Recognized keys:
                                       - 'summary_max_words' (int): Maximum number of words for the summary.
                                                                    Defaults to 100.
                                       - 'summary_ratio' (float): Ratio of original text words to keep (e.g., 0.25 for 25%).
                                                                  Defaults to 0.25.
                                       The effective summary length will be the minimum of these two settings.

        Returns:
            str: A simulated summarized version of the input text. If the input is
                 an empty string, an empty string is returned.

        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input type for TextSummarizerNode. "
                f"Expected str, but received {type(data).__name__}. Data: {data!r}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not data.strip():
            logger.info("Received an empty or whitespace-only string for summarization. Returning an empty string.")
            return ""

        # --- Configuration for summarization length ---
        # Default maximum words for the summary
        default_max_words = 100
        # Default ratio of original text words to keep
        default_summary_ratio = 0.25  # 25% of the original text

        max_words = context.get('summary_max_words', default_max_words)
        summary_ratio = context.get('summary_ratio', default_summary_ratio)

        if not isinstance(max_words, int) or max_words <= 0:
            logger.warning(
                f"Invalid 'summary_max_words' in context ({max_words}). "
                f"Falling back to default: {default_max_words}."
            )
            max_words = default_max_words

        if not isinstance(summary_ratio, (float, int)) or not (0 < summary_ratio <= 1):
            logger.warning(
                f"Invalid 'summary_ratio' in context ({summary_ratio}). "
                f"Falling back to default: {default_summary_ratio}."
            )
            summary_ratio = default_summary_ratio

        words = data.split()
        num_original_words = len(words)

        # Calculate target length based on ratio and absolute maximum
        target_word_count = min(int(num_original_words * summary_ratio), max_words)

        if num_original_words <= target_word_count:
            # If the text is already short enough, return it as is (after stripping leading/trailing whitespace)
            summarized_text = ' '.join(words)
            logger.debug(
                f"Input text already short enough ({num_original_words} words) "
                f"or shorter than target ({target_word_count} words). Returning original."
            )
            return summarized_text.strip()
        else:
            # Simulate summarization by truncating and appending an ellipsis
            summarized_words = words[:target_word_count]
            summarized_text = ' '.join(summarized_words) + "..."
            logger.debug(
                f"Summarized text from {num_original_words} words to "
                f"{len(summarized_words)} words plus ellipsis."
            )
            return summarized_text.strip()