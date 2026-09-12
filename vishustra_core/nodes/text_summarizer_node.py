import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to generate a summary of input text.

    This node accepts a string as input and produces a concise summary.
    The summarization logic is currently simulated based on a configurable
    length reduction ratio, serving as a placeholder for integration with
    advanced summarization models (e.g., abstractive or extractive LLM-based
    summarizers).
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to produce a text summary.

        The `data` is expected to be a string containing the text to be summarized.
        The `context` dictionary can optionally specify `summary_length_ratio`
        to control the output summary's length relative to the original text.

        Args:
            data (Any): The input text intended for summarization. Must be a string.
            context (Dict[str, Any]): A dictionary of operational parameters.
                - `summary_length_ratio` (float, optional): A value between 0.1 and 0.9
                  indicating the desired proportion of the original text's word count
                  for the summary. Defaults to 0.3 if not provided or invalid.

        Returns:
            str: The generated summary of the input text.

        Raises:
            TypeError: If `data` is not a string.
            ValueError: If `data` is an empty string or contains only whitespace.
        """
        if not isinstance(data, str):
            logger.error("TextSummarizerNode received non-string data of type: %s", type(data).__name__)
            raise TypeError(
                f"TextSummarizerNode expects string input for summarization, "
                f"but received type {type(data).__name__}."
            )

        if not data.strip():
            logger.warning("TextSummarizerNode received an empty or whitespace-only string for summarization.")
            raise ValueError("Input text for summarization cannot be empty.")

        # Retrieve and validate summary_length_ratio from context
        summary_length_ratio = context.get("summary_length_ratio", 0.3)
        if not isinstance(summary_length_ratio, (int, float)) or not (0.1 <= summary_length_ratio <= 0.9):
            logger.warning(
                "Invalid 'summary_length_ratio' value '%s' in context. Expected float between 0.1 and 0.9. "
                "Defaulting to 0.3.",
                summary_length_ratio
            )
            summary_length_ratio = 0.3

        original_words = data.split()
        original_word_count = len(original_words)

        # Handle very short texts by returning them as-is
        if original_word_count <= 10: # Heuristic: if text is 10 words or less, return full text
            logger.info(
                "Input text is very short (%d words). Returning full text as summary.",
                original_word_count
            )
            return data.strip()

        # Calculate target summary length, ensuring a minimum and not exceeding original length
        target_word_count = max(int(original_word_count * summary_length_ratio), 5) # Minimum 5 words
        target_word_count = min(target_word_count, original_word_count)

        if target_word_count == original_word_count:
            logger.debug(
                "Calculated target summary word count (%d) equals original (%d). Returning full text.",
                target_word_count, original_word_count
            )
            return data.strip()

        # Simulate summarization by truncating the text
        summary_words = original_words[:target_word_count]
        summary_text = " ".join(summary_words)

        # Append an ellipsis to indicate truncation, if applicable
        if len(summary_words) < original_word_count:
            summary_text += "..."

        logger.info(
            "Summarized text from %d words to approximately %d words (ratio: %.2f).",
            original_word_count, len(summary_words), summary_length_ratio
        )
        return summary_text.strip()