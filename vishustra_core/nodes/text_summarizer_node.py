import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that summarizes text input.

    This node takes a string as input and returns a condensed version
    based on a configurable maximum number of sentences and a minimum
    character length. It simulates summarization by extracting the
    initial sentences of the input text.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a summary.

        Args:
            data: The input text as a string to be summarized.
            context: A dictionary containing operational context,
                     potentially including 'summary_config' with:
                     - 'max_sentences' (int): The maximum number of sentences
                       to include in the summary. Defaults to 5 if not provided
                       or invalid. Must be a positive integer.
                     - 'min_length_chars' (int): The minimum character length
                       the summary should aim for. Defaults to 50 if not provided
                       or invalid. Must be a non-negative integer.

        Returns:
            A string containing the summarized text. Returns an empty string
            if the input text is empty or cannot be summarized.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping whitespace.
        """
        if not isinstance(data, str):
            logger.error(
                "Invalid input data type for TextSummarizerNode. Expected string, got %s.",
                type(data).__name__
            )
            raise TypeError(f"TextSummarizerNode expects string input, but received {type(data).__name__}.")

        stripped_data = data.strip()
        if not stripped_data:
            logger.error("TextSummarizerNode received an empty string for summarization after stripping whitespace.")
            raise ValueError("TextSummarizerNode received an empty string for summarization.")

        # Extract configuration from context, with robust defaults
        summary_config = context.get('summary_config', {})
        
        max_sentences = summary_config.get('max_sentences', 5)
        if not isinstance(max_sentences, int) or max_sentences <= 0:
            logger.warning(
                "Invalid 'max_sentences' configured in context (%s). Falling back to default: 5.",
                max_sentences
            )
            max_sentences = 5
        
        min_length_chars = summary_config.get('min_length_chars', 50)
        if not isinstance(min_length_chars, int) or min_length_chars < 0:
            logger.warning(
                "Invalid 'min_length_chars' configured in context (%s). Falling back to default: 50.",
                min_length_chars
            )
            min_length_chars = 50

        # Simple sentence tokenization (a more advanced NLP library would be used in production)
        # Splits by common sentence terminators followed by whitespace.
        sentences = re.split(r'(?<=[.!?])\s+', stripped_data)
        sentences = [s.strip() for s in sentences if s.strip()] # Clean up empty strings

        if not sentences:
            logger.warning("No sentences could be extracted from the input text in TextSummarizerNode.")
            return ""

        summary_sentences = []
        current_summary_length = 0

        # Build summary, prioritizing max_sentences while also trying to meet min_length_chars
        for i, sentence in enumerate(sentences):
            # Always add up to max_sentences
            if i < max_sentences:
                summary_sentences.append(sentence)
                current_summary_length += len(sentence) + 1 # +1 for space between sentences
            # If max_sentences is reached but min_length_chars isn't met, continue adding
            elif current_summary_length < min_length_chars:
                summary_sentences.append(sentence)
                current_summary_length += len(sentence) + 1
            else:
                break # Both conditions met, or max_sentences reached and min_length_chars not required further

        summary = " ".join(summary_sentences).strip()

        # Final check to ensure min_length_chars is met if possible and not all sentences were used
        if len(summary) < min_length_chars and len(summary_sentences) < len(sentences):
            logger.debug(
                "Current summary length (%d) is less than 'min_length_chars' (%d). "
                "Attempting to extend summary.",
                len(summary), min_length_chars
            )
            # Add remaining sentences until min_length_chars is met or all sentences are used
            for sentence_to_add in sentences[len(summary_sentences):]:
                summary_sentences.append(sentence_to_add)
                summary = " ".join(summary_sentences).strip()
                if len(summary) >= min_length_chars:
                    break
        
        logger.info(
            "Text summarized using %d sentences (max_sentences=%d, min_length_chars=%d). "
            "Original length: %d chars, Summary length: %d chars.",
            len(summary_sentences), max_sentences, min_length_chars,
            len(data), len(summary)
        )

        return summary