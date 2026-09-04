import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates abstractive text summarization.
    It takes a string of text and returns a condensed version, mimicking
    the behavior of an LLM-based summarization without actual model inference.
    """

    def __init__(self, max_summary_length_words: int = 75):
        """
        Initializes the TextSummarizerNode with configuration for summary length.

        Args:
            max_summary_length_words (int): The maximum number of words
                                            to aim for in the simulated summary.
                                            Must be a positive integer.
        Raises:
            ValueError: If `max_summary_length_words` is not a positive integer.
        """
        if not isinstance(max_summary_length_words, int) or max_summary_length_words <= 0:
            logger.error(f"Invalid max_summary_length_words: {max_summary_length_words}. Must be a positive integer.")
            raise ValueError("max_summary_length_words must be a positive integer.")
        self._max_summary_length_words = max_summary_length_words
        logger.debug(f"TextSummarizerNode initialized with max_summary_length_words={self._max_summary_length_words}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by simulating text summarization.

        This method expects `data` to be a string. It performs a basic
        sentence-level truncation to simulate a summary, aiming for
        a specified maximum word count.

        Args:
            data (Any): The input text to be summarized.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly utilized
                                       by this node's core logic but available
                                       for broader orchestration.

        Returns:
            str: The simulated summarized text. If the original text is shorter
                 than the target summary length, the original text is returned.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty or whitespace-only string.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data of type: {type(data).__name__}")
            raise TypeError(f"Input data for TextSummarizerNode must be a string, but received {type(data).__name__}")

        trimmed_data = data.strip()
        if not trimmed_data:
            logger.warning("TextSummarizerNode received an empty or whitespace-only string. Returning an empty string.")
            return ""

        original_words = trimmed_data.split()
        if len(original_words) <= self._max_summary_length_words:
            logger.debug(f"Original text is shorter than or equal to max_summary_length_words ({self._max_summary_length_words}). Returning original text.")
            return trimmed_data

        logger.info(f"Summarizing text of length {len(trimmed_data)} characters, targeting ~{self._max_summary_length_words} words.")

        # Simulate summarization by taking the first few sentences/words
        # This is a rudimentary approach; actual LLM-based summarization
        # would involve more complex understanding and generation.
        sentences = re.split(r'(?<=[.!?])\s+', trimmed_data)
        summary_parts = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if current_word_count + len(sentence_words) <= self._max_summary_length_words:
                summary_parts.append(sentence)
                current_word_count += len(sentence_words)
            else:
                # If adding the whole sentence exceeds the limit, and we haven't added anything yet,
                # take a truncated portion of this sentence to meet the minimum summary.
                if not summary_parts:
                    truncated_words = sentence_words[:self._max_summary_length_words]
                    summary_parts.append(" ".join(truncated_words))
                    current_word_count = len(truncated_words)
                break # Stop processing further sentences

        simulated_summary = " ".join(summary_parts).strip()

        # Add ellipsis if the summary truly truncated the original text and doesn't end naturally
        if len(simulated_summary.split()) < len(original_words) and simulated_summary and not re.search(r'[.!?]$', simulated_summary):
             simulated_summary += "..."

        logger.debug(f"Successfully simulated summary. Original word count: {len(original_words)}, Summary word count: {len(simulated_summary.split())}")
        return simulated_summary