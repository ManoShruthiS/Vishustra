import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates abstractive text summarization.

    This node takes a string of text as input and produces a shorter summary.
    It can be configured with parameters like desired summary length and minimum
    input text length via the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input text to generate a summary.

        Args:
            data: The input text to be summarized (expected to be a string).
            context: A dictionary containing operational parameters for summarization,
                     such as `max_summary_words` and `min_input_chars`.

        Returns:
            A string representing the summarized text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `max_summary_words` or `min_input_chars` in context
                        are non-positive integers.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data)}")
            raise TypeError("Input data for TextSummarizerNode must be a string.")

        # Default configuration parameters
        DEFAULT_MAX_SUMMARY_WORDS = 50
        DEFAULT_MIN_INPUT_CHARS = 150

        # Retrieve parameters from context with robust error handling
        max_summary_words = context.get('max_summary_words', DEFAULT_MAX_SUMMARY_WORDS)
        min_input_chars = context.get('min_input_chars', DEFAULT_MIN_INPUT_CHARS)

        if not isinstance(max_summary_words, int) or max_summary_words <= 0:
            logger.warning(
                f"Invalid 'max_summary_words' in context ({max_summary_words}). "
                f"Using default: {DEFAULT_MAX_SUMMARY_WORDS}."
            )
            max_summary_words = DEFAULT_MAX_SUMMARY_WORDS
        
        if not isinstance(min_input_chars, int) or min_input_chars <= 0:
            logger.warning(
                f"Invalid 'min_input_chars' in context ({min_input_chars}). "
                f"Using default: {DEFAULT_MIN_INPUT_CHARS}."
            )
            min_input_chars = DEFAULT_MIN_INPUT_CHARS

        input_text = data.strip()

        if not input_text:
            logger.info("TextSummarizerNode received empty input text. Returning empty string.")
            return ""

        if len(input_text) < min_input_chars:
            logger.warning(
                f"Input text length ({len(input_text)} chars) is below "
                f"minimum required ({min_input_chars} chars) for summarization. "
                "Returning original text without summarization."
            )
            return input_text

        # Simulate summarization: Take sentences until desired word count is reached.
        # This is a basic simulation and not an actual NLP summarizer.
        sentences = re.split(r'(?<=[.!?])\s+', input_text)
        
        summarized_parts = []
        current_word_count = 0

        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            if current_word_count + words_in_sentence <= max_summary_words:
                summarized_parts.append(sentence)
                current_word_count += words_in_sentence
            else:
                # If adding the whole sentence exceeds max_summary_words,
                # try to truncate the current sentence if it's the first part,
                # or just stop if we already have some content.
                if not summarized_parts: # If no sentences added yet, try to truncate the first one
                    truncated_words = sentence.split()[:max_summary_words]
                    if truncated_words:
                        summarized_parts.append(" ".join(truncated_words))
                        current_word_count = len(truncated_words)
                break # Stop adding more sentences

        summary = " ".join(summarized_parts).strip()

        if not summary and input_text:
            # Fallback for very short text where sentence splitting might be problematic
            # or if the first sentence itself is too long for max_summary_words
            logger.debug(
                "Generated summary is empty despite non-empty input. "
                "Attempting fallback by truncating words."
            )
            words = input_text.split()
            summary = " ".join(words[:max_summary_words])

        logger.info(
            f"TextSummarizerNode successfully summarized text. "
            f"Original length: {len(input_text)} chars. "
            f"Summary length: {len(summary)} chars ({len(summary.split())} words)."
        )
        return summary