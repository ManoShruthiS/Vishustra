import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    This node takes a string as input and returns a simulated summary
    of that text. The summarization logic is intentionally simple for
    demonstration purposes, often truncating the text based on word count.

    Context parameters that can influence behavior:
    - 'summarizer_target_words' (int, default: 50): The approximate maximum
      number of words for the summary.
    - 'summarizer_min_original_length' (int, default: 100): If the original
      text has fewer words than this, it might be returned without summarization,
      unless it's still longer than `summarizer_target_words`.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data by generating a simulated summary.

        Args:
            data (Any): The input data, expected to be a string representing the text
                        to be summarized.
            context (Dict[str, Any]): A dictionary containing additional context or
                                      configuration parameters for the node.
                                      Can include 'summarizer_target_words' and
                                      'summarizer_min_original_length'.

        Returns:
            Any: The simulated summarized text (a string).

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Input data must be a string for summarization. "
                f"Received type: {type(data).__name__}. Data: {data!r}"
            )
            raise TypeError(
                f"Invalid input type for {self.node_name}. Expected str, "
                f"got {type(data).__name__}."
            )

        text_to_summarize = data.strip()
        if not text_to_summarize:
            logger.warning(
                f"[{self.node_name}] Received an empty string for summarization. "
                f"Returning empty string."
            )
            return ""

        # Retrieve summarization parameters from context or use sensible defaults
        target_word_count = context.get("summarizer_target_words", 50)
        min_original_length_for_summary = context.get("summarizer_min_original_length", 100)

        if not isinstance(target_word_count, int) or target_word_count <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'summarizer_target_words' in context "
                f"({target_word_count!r}). Using default of 50."
            )
            target_word_count = 50
        
        if not isinstance(min_original_length_for_summary, int) or min_original_length_for_summary < 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'summarizer_min_original_length' in context "
                f"({min_original_length_for_summary!r}). Using default of 100."
            )
            min_original_length_for_summary = 100


        words = text_to_summarize.split()
        original_word_count = len(words)

        if original_word_count <= target_word_count:
            # If the text is already shorter than or equal to the target, no actual summarization is needed.
            logger.debug(
                f"[{self.node_name}] Original text ({original_word_count} words) "
                f"is already within or below target_word_count ({target_word_count}). "
                f"Returning full text."
            )
            return text_to_summarize
        
        if original_word_count <= min_original_length_for_summary:
            # If the text is considered "short" based on min_original_length_for_summary,
            # but still longer than the target_word_count, we still summarize.
            # This is to handle cases where min_original_length is larger than target.
            logger.debug(
                f"[{self.node_name}] Original text ({original_word_count} words) "
                f"is considered short but still longer than target. "
                f"Applying summarization."
            )


        # Simulate summarization by truncating to target_word_count words
        summary_words = words[:target_word_count]
        summarized_text = " ".join(summary_words) + "..."

        logger.info(
            f"[{self.node_name}] Summarized text from {original_word_count} words "
            f"to approximately {len(summary_words)} words (target: {target_word_count})."
        )
        return summarized_text