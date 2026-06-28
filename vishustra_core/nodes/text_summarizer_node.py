import logging
import re
from typing import Any, Dict

# Assuming BaseNode is correctly available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.

    This node takes a string as input and returns a summarized version
    based on configuration parameters provided in the context. The
    summarization here is a heuristic-based approximation, primarily
    for demonstrating node functionality and integration within Vishustra,
    rather than a sophisticated NLP model. It truncates the text to
    a specified word count or ratio, respecting minimum and maximum length constraints.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input text to generate a summary.

        Args:
            data (Any): The input data to be summarized. Expected to be a string
                        containing the text content.
            context (Dict[str, Any]): A dictionary containing configuration for
                                       the summarization process. Supported keys:
                                       - 'summary_ratio' (float, optional): The target ratio
                                         of the original text's word count to retain in the
                                         summary. Defaults to 0.3 (30%). Must be between 0.0 and 1.0.
                                       - 'min_words' (int, optional): The absolute minimum number
                                         of words the summary should contain. Defaults to 20.
                                       - 'max_words' (int, optional): The absolute maximum number
                                         of words the summary should contain. Defaults to 150.
                                         If 0, no upper limit is applied (beyond the original text).

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summary_ratio' is out of bounds or other context
                        parameters are invalid after type checking.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, but received {type(data).__name__}."
            )

        text = data.strip()
        if not text:
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only text for summarization. Returning empty string."
            )
            return ""

        # Default summarization parameters
        DEFAULT_SUMMARY_RATIO = 0.3
        DEFAULT_MIN_WORDS = 20
        DEFAULT_MAX_WORDS = 150 # 0 means no upper limit from max_words, rely on ratio/min

        # Retrieve and validate summary_ratio from context
        summary_ratio = context.get("summary_ratio", DEFAULT_SUMMARY_RATIO)
        if not isinstance(summary_ratio, (int, float)) or not (0.0 <= summary_ratio <= 1.0):
            logger.warning(
                f"[{self.node_name}] Invalid or out-of-range 'summary_ratio' ({summary_ratio}) "
                f"in context. Falling back to default: {DEFAULT_SUMMARY_RATIO}."
            )
            summary_ratio = DEFAULT_SUMMARY_RATIO

        # Retrieve and validate min_words from context
        min_words = context.get("min_words", DEFAULT_MIN_WORDS)
        if not isinstance(min_words, int) or min_words < 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'min_words' ({min_words}) in context. "
                f"Falling back to default: {DEFAULT_MIN_WORDS}."
            )
            min_words = DEFAULT_MIN_WORDS

        # Retrieve and validate max_words from context
        max_words = context.get("max_words", DEFAULT_MAX_WORDS)
        if not isinstance(max_words, int) or max_words < 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'max_words' ({max_words}) in context. "
                f"Falling back to default: {DEFAULT_MAX_WORDS}."
            )
            max_words = DEFAULT_MAX_WORDS
        
        # Ensure min_words does not exceed max_words, unless max_words is 0 (unlimited)
        if max_words != 0 and min_words > max_words:
            logger.warning(
                f"[{self.node_name}] 'min_words' ({min_words}) is greater than 'max_words' ({max_words}). "
                f"Adjusting 'max_words' to be equal to 'min_words'."
            )
            max_words = min_words

        # Split text into words, considering common delimiters.
        # This is a basic split; a real NLP pipeline would use tokenization.
        words = re.findall(r'\b\w+\b', text.lower())
        original_word_count = len(words)

        if original_word_count <= min_words:
            logger.info(
                f"[{self.node_name}] Original text is shorter than or equal to 'min_words' ({min_words}). "
                f"Returning full text."
            )
            return text

        # Calculate target word count based on ratio
        target_word_count = int(original_word_count * summary_ratio)

        # Apply min/max constraints
        final_word_count = target_word_count
        if min_words > 0:
            final_word_count = max(min_words, final_word_count)
        if max_words > 0:
            final_word_count = min(max_words, final_word_count)
        
        # Ensure final_word_count does not exceed original text length
        final_word_count = min(original_word_count, final_word_count)

        if final_word_count >= original_word_count:
            logger.info(
                f"[{self.node_name}] Calculated summary length ({final_word_count} words) is "
                f"greater than or equal to original text length ({original_word_count} words). "
                f"Returning full text."
            )
            return text

        # Find the actual text segment corresponding to `final_word_count` words.
        # This is a simplified approach, merely truncating words.
        current_word_count = 0
        summary_end_index = 0
        for match in re.finditer(r'\b\w+\b', text):
            current_word_count += 1
            summary_end_index = match.end()
            if current_word_count == final_word_count:
                break
        
        summarized_text = text[:summary_end_index].strip()

        # Add an ellipsis if the text was truncated and doesn't end naturally.
        if final_word_count < original_word_count and not summarized_text.endswith(('.', '!', '?', '...')):
             summarized_text += "..."

        logger.info(
            f"[{self.node_name}] Summarized text from {original_word_count} words to "
            f"{current_word_count} words (target ratio: {summary_ratio}, "
            f"min: {min_words}, max: {max_words})."
        )

        return summarized_text

