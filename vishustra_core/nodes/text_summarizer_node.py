from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict
import logging
import re

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates abstractive text summarization.
    It takes a text string and attempts to produce a shorter,
    condensed version based on configured parameters.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a summary.

        The summarization logic can be controlled via the 'context' dictionary:
        - 'summary_ratio': A float between 0.0 and 1.0 representing the
                           desired ratio of the summary's length to the
                           original text's length. Defaults to 0.3 (30%).
                           Applied on sentence count.
        - 'max_sentences': An integer specifying the maximum number of
                           sentences to include in the summary. Overrides
                           'summary_ratio' if present.
        - 'min_sentences': An integer specifying the minimum number of
                           sentences for the summary. Defaults to 1.

        Args:
            data: The input text to be summarized. Expected to be a string.
            context: A dictionary containing parameters for summarization,
                     such as 'summary_ratio', 'max_sentences', or 'min_sentences'.

        Returns:
            A string containing the summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summary_ratio' or 'max_sentences' in context are invalid.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data)}. Expected str.")
            raise TypeError("TextSummarizerNode requires string input for summarization.")

        if not data.strip():
            logger.warning("TextSummarizerNode received empty or whitespace-only data. Returning empty string.")
            return ""

        # Default summarization parameters
        default_summary_ratio = 0.3
        default_max_sentences = 5
        default_min_sentences = 1

        summary_ratio = context.get('summary_ratio', default_summary_ratio)
        max_sentences_config = context.get('max_sentences')
        min_sentences_config = context.get('min_sentences', default_min_sentences)

        if not (0.0 <= summary_ratio <= 1.0):
            logger.error(f"Invalid 'summary_ratio' in context: {summary_ratio}. Must be between 0.0 and 1.0.")
            raise ValueError("Context 'summary_ratio' must be between 0.0 and 1.0.")
        
        if not isinstance(min_sentences_config, int) or min_sentences_config < 0:
            logger.error(f"Invalid 'min_sentences' in context: {min_sentences_config}. Must be a non-negative integer.")
            raise ValueError("Context 'min_sentences' must be a non-negative integer.")

        # A simple approach to split into sentences, handling common delimiters
        # This is a simulation; real NLP would use more sophisticated tokenizers.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', data) if s.strip()]
        original_sentence_count = len(sentences)

        if original_sentence_count == 0:
            return "" # No sentences found

        target_sentence_count = 0
        if max_sentences_config is not None:
            if not isinstance(max_sentences_config, int) or max_sentences_config < 0:
                logger.error(f"Invalid 'max_sentences' in context: {max_sentences_config}. Must be a non-negative integer.")
                raise ValueError("Context 'max_sentences' must be a non-negative integer.")
            
            target_sentence_count = min(max_sentences_config, original_sentence_count)
            logger.debug(f"Using 'max_sentences' from context: {max_sentences_config}. Target sentences: {target_sentence_count}")
        else:
            target_sentence_count = int(original_sentence_count * summary_ratio)
            logger.debug(f"Using 'summary_ratio' from context: {summary_ratio}. Target sentences: {target_sentence_count}")

        # Ensure we always return at least min_sentences, but not more than original_sentence_count
        final_sentence_count = max(min_sentences_config, min(target_sentence_count, original_sentence_count))
        
        # Take the first 'final_sentence_count' sentences for simplicity in simulation
        summarized_sentences = sentences[:final_sentence_count]
        summary = " ".join(summarized_sentences)

        original_length = len(data)
        summary_length = len(summary)

        logger.info(
            f"Summarized text from {original_sentence_count} sentences ({original_length} chars) "
            f"to {final_sentence_count} sentences ({summary_length} chars) "
            f"using '{'max_sentences' if max_sentences_config is not None else 'summary_ratio'}'."
        )

        return summary

