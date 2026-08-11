import logging
from typing import Any, Dict
import re

# Assuming BaseNode is correctly located at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to simulate text summarization.

    This node takes a string as input and produces a shorter summary.
    The summarization logic is simplified for simulation purposes, primarily
    extracting key sentences from the beginning of the text until a
    configurable maximum word count is reached.
    """

    def __init__(self, default_max_summary_words: int = 100):
        """
        Initializes the TextSummarizerNode with default configuration.

        Args:
            default_max_summary_words (int): The default maximum number of words
                                             for the summary if not explicitly provided
                                             in the process context. Must be a positive integer.
        Raises:
            ValueError: If `default_max_summary_words` is not a positive integer.
        """
        if not isinstance(default_max_summary_words, int) or default_max_summary_words <= 0:
            raise ValueError("`default_max_summary_words` must be a positive integer.")
        self._default_max_summary_words = default_max_summary_words
        logger.debug(
            f"TextSummarizerNode initialized with default_max_summary_words: {default_max_summary_words}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by generating a summary of the text.

        The method expects `data` to be a string. The `context` dictionary
        can override the default summary length using the 'max_summary_words' key.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary of runtime parameters.
                                      Optional key: 'max_summary_words' (int) to
                                      specify the desired maximum word count for the summary.

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty or whitespace-only string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for {self.node_name}. Expected `str`, got `{type(data).__name__}`."
            )
            raise TypeError(
                f"{self.node_name} expects string input, but received `{type(data).__name__}`."
            )

        original_text = data.strip()
        if not original_text:
            logger.warning(
                f"Received empty or whitespace-only string for summarization in {self.node_name}."
            )
            raise ValueError("Cannot summarize an empty or whitespace-only string.")

        original_words = original_text.split()
        original_word_count = len(original_words)
        logger.info(f"Initiating summarization for text of ~{original_word_count} words.")

        # Determine max_summary_words, prioritizing context over instance default
        max_summary_words = context.get("max_summary_words", self._default_max_summary_words)
        if not isinstance(max_summary_words, int) or max_summary_words <= 0:
            logger.warning(
                f"Invalid or non-positive 'max_summary_words' in context ({max_summary_words}). "
                f"Falling back to default of {self._default_max_summary_words}."
            )
            max_summary_words = self._default_max_summary_words

        # If the original text is already concise (e.g., less than 1.5 times the target length),
        # return it as is to avoid unnecessary truncation or overly short summaries.
        if original_word_count <= max_summary_words * 1.5:
            logger.debug(
                f"Original text (~{original_word_count} words) is short relative to target "
                f"({max_summary_words} words). Returning full text."
            )
            return original_text

        # Simple sentence tokenization: splits by common sentence-ending punctuation
        # followed by one or more whitespace characters.
        # This regex attempts to avoid splitting on common abbreviations (e.g., "Dr.", "U.S.").
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", original_text)

        # Fallback to simpler split if the initial split yields too few sentences,
        # which can happen with complex or malformed text.
        if len(sentences) <= 1:
            sentences = re.split(r"(?<=[.!?])\s+", original_text)

        summary_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words_list = sentence.split()
            sentence_word_count = len(sentence_words_list)

            # Check if adding this sentence would exceed the max word count
            if current_word_count + sentence_word_count <= max_summary_words:
                summary_sentences.append(sentence)
                current_word_count += sentence_word_count
            elif current_word_count == 0 and sentence_word_count > max_summary_words:
                # If the very first sentence is longer than the target,
                # truncate it and add an ellipsis.
                truncated_sentence = " ".join(sentence_words_list[:max_summary_words])
                summary_sentences.append(truncated_sentence + "...")
                current_word_count = max_summary_words
                break  # Stop, as we've hit the limit with just one sentence
            else:
                break  # Stop adding sentences if the next one would exceed the limit

        if not summary_sentences:
            # This can occur if the sentence splitting was ineffective for very unusual text,
            # or if the text was too short to yield meaningful sentences but still longer
            # than the `max_summary_words * 1.5` threshold.
            logger.warning(
                f"No sentences were added to the summary. Attempting simple word-level truncation."
            )
            if original_word_count > max_summary_words:
                return " ".join(original_words[:max_summary_words]) + "..."
            else:
                return original_text  # Should theoretically be caught by the earlier check

        summary = " ".join(summary_sentences).strip()
        summary_word_count = len(summary.split())
        logger.info(
            f"Summarization complete. Original words: {original_word_count}, "
            f"Summary words: {summary_word_count} (target: {max_summary_words})."
        )

        return summary