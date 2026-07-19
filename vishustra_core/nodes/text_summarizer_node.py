import logging
from typing import Any, Dict

# Assuming vishustra_core is a properly installed package in the environment
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that summarizes input text.

    This node provides a simulated summarization, condensing text based on
    either a specified word count ratio or a hard limit on the number of words.
    It's designed for quick content condensation within workflows.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by summarizing it.

        The summarization logic is a basic simulation: it extracts a portion
        of the input text from the beginning, based on configured parameters
        in the `context`.

        Args:
            data: The input content to be summarized (expected to be a string).
            context: A dictionary containing parameters for summarization.
                     Supported keys:
                     - 'max_words' (int): The maximum number of words to retain
                       in the summary. If provided and valid (non-negative),
                       this parameter takes precedence.
                     - 'summary_ratio' (float): The ratio of the original text's
                       word count to include in the summary (e.g., 0.3 for 30%).
                       Used if 'max_words' is not specified or invalid.
                       Must be between 0.0 and 1.0. Defaults to 0.3.

        Returns:
            str: The condensed text, potentially with an ellipsis appended
                 if truncation occurred. Returns an empty string for empty input.

        Raises:
            ValueError: If the input 'data' is not of type string.
            Exception: Propagates unexpected errors during processing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'."
            )
            raise ValueError(
                f"Input 'data' for '{self.node_name}' must be a string, but received '{type(data).__name__}'."
            )

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only input string. Returning empty string.")
            return ""

        words = data.split()
        original_word_count = len(words)
        summary_word_count: int

        # Attempt to use 'max_words' from context first
        max_words_config = context.get("max_words")
        if isinstance(max_words_config, int) and max_words_config >= 0:
            summary_word_count = min(original_word_count, max_words_config)
            logger.debug(
                f"[{self.node_name}] Summarizing to a maximum of {max_words_config} words. Original: {original_word_count} words."
            )
        else:
            # Fallback to 'summary_ratio'
            summary_ratio_config = context.get("summary_ratio", 0.3)
            if not isinstance(summary_ratio_config, (int, float)) or not (0.0 <= summary_ratio_config <= 1.0):
                logger.warning(
                    f"[{self.node_name}] Invalid or out-of-range 'summary_ratio' ({summary_ratio_config}) in context. Using default ratio of 0.3."
                )
                summary_ratio_config = 0.3

            summary_word_count = int(original_word_count * summary_ratio_config)
            logger.debug(
                f"[{self.node_name}] Summarizing using ratio {summary_ratio_config}. Original: {original_word_count} words, Target: {summary_word_count} words."
            )

        # Ensure at least one word is returned for non-empty input, unless target is explicitly 0
        if summary_word_count == 0 and original_word_count > 0:
            summary_word_count = 1
            logger.debug(f"[{self.node_name}] Adjusted summary word count to 1 to prevent empty summary for non-empty input.")

        summarized_words = words[:summary_word_count]
        summarized_text = " ".join(summarized_words)

        if original_word_count > len(summarized_words):
            # Append ellipsis if the text was actually truncated
            summarized_text += "..."

        logger.info(
            f"[{self.node_name}] Successfully summarized text from {original_word_count} words to {len(summarized_words)} words."
        )
        return summarized_text