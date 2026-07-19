import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.

    This node takes a string of text and generates a shorter version
    based on specified parameters in the context, such as a target
    sentence count or a summary ratio. It is intended for orchestrating
    LLM workflows by providing a modular component for text reduction.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a summary.

        The summarization is simulated by extracting a subset of sentences
        from the input text.

        Expected 'data':
            A string containing the text to be summarized.

        Expected 'context' parameters (optional):
            - 'target_sentence_count' (int): The desired maximum number of
              sentences in the summary. If 0, an empty string is returned.
              Overrides 'summary_ratio' if both are provided.
            - 'summary_ratio' (float): The ratio of original sentences to
              retain in the summary (e.g., 0.3 for 30%). If this results in
              zero sentences but the original text has content, at least
              one sentence will be returned (unless explicitly 0.0 ratio).
              Must be between 0.0 and 1.0.

        Returns:
            str: The summarized text. An empty string is returned if the
                 input data is empty, no sentences are detected, or if
                 'target_sentence_count' is explicitly 0.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_sentence_count' or 'summary_ratio' in the
                        context are of invalid types or out of acceptable ranges.
        """
        if not isinstance(data, str):
            logger.error(
                f"TextSummarizerNode received invalid data type. Expected str, got {type(data).__name__}."
            )
            raise TypeError(
                f"Input 'data' must be a string, got {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(
                "TextSummarizerNode received empty or whitespace-only text. Returning empty string."
            )
            return ""

        # Simple sentence tokenizer for simulation purposes.
        # This regex splits by common sentence terminators (., !, ?) followed by a space,
        # keeping the terminator as part of the preceding sentence.
        sentences = re.split(r'(?<=[.!?])\s+', data.strip())
        sentences = [s.strip() for s in sentences if s.strip()] # Filter out any empty strings resulting from split

        if not sentences:
            logger.warning("TextSummarizerNode could not detect any sentences in the input. Returning empty string.")
            return ""

        target_sentence_count = context.get('target_sentence_count')
        summary_ratio = context.get('summary_ratio')

        num_sentences_to_keep = 0 # Initialize, will be updated based on context or defaults

        if target_sentence_count is not None:
            if not isinstance(target_sentence_count, int) or target_sentence_count < 0:
                logger.error(
                    f"Invalid 'target_sentence_count' in context: {target_sentence_count}. "
                    "Must be a non-negative integer."
                )
                raise ValueError(
                    f"'target_sentence_count' must be a non-negative integer, "
                    f"got {type(target_sentence_count).__name__}."
                )
            num_sentences_to_keep = min(target_sentence_count, len(sentences))
        elif summary_ratio is not None:
            if not isinstance(summary_ratio, (int, float)) or not (0.0 <= summary_ratio <= 1.0):
                logger.error(
                    f"Invalid 'summary_ratio' in context: {summary_ratio}. "
                    "Must be a float between 0.0 and 1.0."
                )
                raise ValueError(
                    f"'summary_ratio' must be a float between 0.0 and 1.0, "
                    f"got {type(summary_ratio).__name__}."
                )
            # Calculate based on ratio. Ensure at least 1 sentence if original has sentences and ratio > 0,
            # unless the ratio itself is 0.0.
            if len(sentences) > 0 and summary_ratio > 0.0:
                num_sentences_to_keep = max(1, int(len(sentences) * summary_ratio))
            num_sentences_to_keep = min(num_sentences_to_keep, len(sentences))
        else:
            # Default behavior if no parameters are provided
            default_summary_ratio = 0.3
            if len(sentences) > 0: # Ensure we don't try to get a sentence from an empty list
                num_sentences_to_keep = max(1, int(len(sentences) * default_summary_ratio))
            num_sentences_to_keep = min(num_sentences_to_keep, len(sentences))
            logger.debug(f"No summarization parameters provided. Using default summary_ratio of {default_summary_ratio}.")

        summarized_text = " ".join(sentences[:num_sentences_to_keep])

        logger.info(
            f"TextSummarizerNode processed text (original length: {len(data)} chars, "
            f"summary length: {len(summarized_text)} chars, "
            f"sentences kept: {num_sentences_to_keep})."
        )
        return summarized_text