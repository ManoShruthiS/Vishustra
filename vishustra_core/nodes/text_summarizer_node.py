import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to condense input text.
    It simulates text summarization based on a configurable ratio,
    providing a shorter version of the original content.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by simulating text summarization.

        This method expects `data` to be a string containing the text to be summarized.
        The `context` dictionary can optionally include:
        - `summary_ratio` (float): The target ratio of the summary length to the
          original text length. For example, `0.3` would aim for a summary
          that is 30% of the original text's word count. Defaults to `0.3`.

        Args:
            data (Any): The text content (expected to be a string) to be summarized.
            context (Dict[str, Any]): A dictionary containing runtime parameters for summarization.

        Returns:
            str: The simulated summarized text. This will be a truncated version of the
                 original text followed by an ellipsis if actual summarization occurred.

        Raises:
            ValueError: If `data` is not a string, or if `summary_ratio` in `context` is
                        not a valid float between 0 (exclusive) and 1 (inclusive).
            RuntimeError: For unexpected errors during the summarization process.
        """
        if not isinstance(data, str):
            logger.error("TextSummarizerNode received non-string data. Type: %s", type(data))
            raise ValueError(f"TextSummarizerNode expects string input for summarization, but received {type(data)}.")

        if not data.strip():
            logger.info("TextSummarizerNode received empty or whitespace-only string. Returning an empty string.")
            return ""

        summary_ratio = context.get("summary_ratio", 0.3)

        if not isinstance(summary_ratio, (int, float)) or not (0 < summary_ratio <= 1):
            logger.error("Invalid 'summary_ratio' value in context: %s. Must be a float between 0 (exclusive) and 1 (inclusive).", summary_ratio)
            raise ValueError(
                f"Invalid 'summary_ratio' in context: {summary_ratio}. "
                "Must be a float value between 0 (exclusive) and 1 (inclusive)."
            )

        try:
            original_words = data.split()
            original_word_count = len(original_words)
            
            # Calculate target word count, ensuring it's at least 1 if the original has words
            target_word_count = max(1, int(original_word_count * summary_ratio)) if original_word_count > 0 else 0

            if target_word_count >= original_word_count:
                logger.debug("Summary ratio (%.2f) results in target length >= original. Returning original text.", summary_ratio)
                return data # No actual summarization needed if ratio is too high

            summarized_words = original_words[:target_word_count]
            
            # Reconstruct the summarized text and append ellipsis to signify truncation
            summarized_text = " ".join(summarized_words) + "..."
            
            logger.debug(
                "Successfully summarized text from %d words to approximately %d words using ratio %.2f.",
                original_word_count, len(summarized_words), summary_ratio
            )
            return summarized_text

        except Exception as e:
            logger.exception("An unexpected error occurred during the text summarization process in TextSummarizerNode.")
            raise RuntimeError(f"Failed to summarize text due to an internal error: {e}") from e