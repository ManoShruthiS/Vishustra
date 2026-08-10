import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    This node takes a string as input data and returns a summarized version.
    The summarization logic is simulated and configurable via the 'context' dictionary,
    allowing control over the length and percentage of the original text.

    Configuration parameters in 'context':
    - 'summary_percentage' (float): The desired percentage of the original text's
      word count for the summary (e.g., 0.3 for 30%). Must be between 0.0 and 1.0.
      Defaults to 0.3.
    - 'min_words' (int): The minimum number of words the summary should contain.
      Defaults to 50.
    - 'max_words' (int): The maximum number of words the summary should contain.
      Defaults to 200.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating text summarization.

        This simulation extracts a portion of the original text based on
        configured parameters.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        to be summarized.
            context (Dict[str, Any]): A dictionary containing node-specific
                                      configuration and runtime context.
                                      Expected keys:
                                      - 'summary_percentage' (float, optional): Percentage of original
                                        word count for summary. Defaults to 0.3. Must be > 0 and <= 1.
                                      - 'min_words' (int, optional): Minimum words in summary. Defaults to 50.
                                        Must be a non-negative integer.
                                      - 'max_words' (int, optional): Maximum words in summary. Defaults to 200.
                                        Must be a positive integer.

        Returns:
            Any: A string representing the summarized text. An ellipsis "..." is
                 appended if the text was actually truncated.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If context parameters for 'min_words' and 'max_words' are
                        invalid (e.g., max_words < min_words).
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Aborting process."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string, "
                f"but received {type(data).__name__}."
            )

        stripped_data = data.strip()
        if not stripped_data:
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only text for summarization. "
                f"Returning an empty string."
            )
            return ""

        # --- Retrieve and validate context parameters with robust defaults ---
        summary_percentage = context.get('summary_percentage', 0.3)
        min_words = context.get('min_words', 50)
        max_words = context.get('max_words', 200)

        if not (isinstance(summary_percentage, (float, int)) and 0.0 < summary_percentage <= 1.0):
            logger.warning(
                f"[{self.node_name}] Invalid 'summary_percentage' in context. "
                f"Expected a float between (0.0, 1.0], received {summary_percentage}. "
                f"Defaulting to 0.3."
            )
            summary_percentage = 0.3

        if not (isinstance(min_words, int) and min_words >= 0):
            logger.warning(
                f"[{self.node_name}] Invalid 'min_words' in context. "
                f"Expected a non-negative integer, received {min_words}. "
                f"Defaulting to 50."
            )
            min_words = 50

        if not (isinstance(max_words, int) and max_words > 0):
            logger.warning(
                f"[{self.node_name}] Invalid 'max_words' in context. "
                f"Expected a positive integer, received {max_words}. "
                f"Defaulting to 200."
            )
            max_words = 200

        if min_words > max_words:
            logger.error(
                f"[{self.node_name}] Configuration error: 'min_words' ({min_words}) "
                f"cannot be greater than 'max_words' ({max_words}). Aborting process."
            )
            raise ValueError(
                f"Invalid context for '{self.node_name}': 'min_words' ({min_words}) "
                f"must be less than or equal to 'max_words' ({max_words})."
            )

        # --- Simulate summarization by word count truncation ---
        words = stripped_data.split()
        original_word_count = len(words)

        if original_word_count == 0:
            logger.info(f"[{self.node_name}] Input text contained no words after stripping. Returning empty string.")
            return ""

        # If the original text is already shorter than the minimum, return it as-is.
        if original_word_count < min_words:
            logger.info(
                f"[{self.node_name}] Input text ({original_word_count} words) is shorter "
                f"than configured 'min_words' ({min_words}). Returning original text."
            )
            return stripped_data

        # Calculate target word count based on percentage
        target_word_count = int(original_word_count * summary_percentage)

        # Clamp the target word count within the defined min_words and max_words
        effective_word_count = max(min_words, min(max_words, target_word_count))

        # Ensure we don't return more words than originally available unless explicitly required
        # (though min_words logic already handles returning original if too short).
        effective_word_count = min(effective_word_count, original_word_count)

        if effective_word_count >= original_word_count:
            logger.info(
                f"[{self.node_name}] Calculated summary length ({effective_word_count} words) "
                f"is longer than or equal to original text ({original_word_count} words). "
                f"Returning original text without truncation."
            )
            return stripped_data

        summarized_words = words[:effective_word_count]
        summarized_text = " ".join(summarized_words)

        # Append ellipsis if the text was genuinely truncated
        if effective_word_count < original_word_count:
            summarized_text += "..."
            logger.debug(
                f"[{self.node_name}] Successfully summarized text from {original_word_count} words "
                f"to {effective_word_count} words."
            )
        else:
            logger.debug(
                f"[{self.node_name}] Summarized text has {effective_word_count} words (no actual truncation performed)."
            )

        return summarized_text.strip()