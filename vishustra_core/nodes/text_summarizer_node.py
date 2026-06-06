import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to summarize text content.

    This node takes a string as input and produces a condensed version
    based on specified summarization parameters provided within the context.
    The current implementation employs a simple extractive method, retaining
    a proportional number of initial sentences.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to produce a summary.

        The summarization logic implemented here is a straightforward extraction
        method, retaining a calculated number of initial sentences from the
        input text. For advanced summarization, a production system would
        typically integrate specialized NLP models or algorithms.

        Args:
            data (Any): The input data to be processed. This node specifically
                        expects a string containing the text to be summarized.
            context (Dict[str, Any]): A dictionary containing configuration
                                      parameters for the summarization process.
                                      Expected keys include:
                                      - "summary_ratio" (float, optional): The desired
                                        target ratio of the original text's sentence count
                                        to retain. Must be between 0 (exclusive) and 1
                                        (inclusive). Defaults to 0.3 (30%).
                                      - "min_sentences" (int, optional): The minimum
                                        number of sentences to ensure in the summary.
                                        Defaults to 1. Must be a non-negative integer.
                                      - "max_sentences" (int, optional): An optional
                                        upper bound on the number of sentences in the
                                        summary. If provided, must be an integer greater
                                        than or equal to `min_sentences`. Defaults to None,
                                        implying no specific upper limit beyond the ratio.

        Returns:
            Any: A string representing the summarized text. An empty string is
                 returned if the input is empty or if no sentences can be
                 extracted or retained.

        Raises:
            ValueError: If the input 'data' is not of type `str`.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received invalid data type. Expected 'str', got '{type(data).__name__}'.")
            raise ValueError("TextSummarizerNode expects string input data.")

        text = data.strip()
        if not text:
            logger.warning("TextSummarizerNode received an empty string for summarization. Returning empty string.")
            return ""

        # --- Extract and validate context parameters ---
        summary_ratio = context.get("summary_ratio", 0.3)
        min_sentences = context.get("min_sentences", 1)
        max_sentences = context.get("max_sentences")

        # Validate 'summary_ratio'
        if not isinstance(summary_ratio, (int, float)) or not (0 < summary_ratio <= 1):
            logger.warning(
                f"Invalid 'summary_ratio' value in context: '{summary_ratio}'. "
                "Expected a float/int between 0 (exclusive) and 1 (inclusive). Using default 0.3."
            )
            summary_ratio = 0.3

        # Validate 'min_sentences'
        if not isinstance(min_sentences, int) or min_sentences < 0:
            logger.warning(
                f"Invalid 'min_sentences' value in context: '{min_sentences}'. "
                "Expected a non-negative integer. Using default 1."
            )
            min_sentences = 1
        # Ensure min_sentences is at least 1 for non-empty text
        min_sentences = max(1, min_sentences)

        # Validate 'max_sentences'
        if max_sentences is not None:
            if not isinstance(max_sentences, int) or max_sentences < min_sentences:
                logger.warning(
                    f"Invalid 'max_sentences' value in context: '{max_sentences}'. "
                    "Expected an integer greater than or equal to 'min_sentences'. Ignoring 'max_sentences'."
                )
                max_sentences = None
        # --- End context parameter validation ---

        # Simple sentence splitting using regex. This is a basic approach for demonstration.
        # For a more robust solution in production, consider NLP libraries like NLTK or spaCy
        # for advanced sentence tokenization, which handles abbreviations and complex punctuation better.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        original_sentence_count = len(sentences)

        if original_sentence_count == 0:
            logger.debug("No sentences identified in the input text for TextSummarizerNode. Returning empty string.")
            return ""

        # Calculate the target number of sentences based on ratio and boundaries
        target_sentence_count_by_ratio = int(original_sentence_count * summary_ratio)
        
        # Apply minimum sentences constraint
        effective_target_sentence_count = max(min_sentences, target_sentence_count_by_ratio)

        # Apply maximum sentences constraint if provided
        if max_sentences is not None:
            effective_target_sentence_count = min(effective_target_sentence_count, max_sentences)

        # Ensure we don't try to take more sentences than are actually available in the text
        effective_target_sentence_count = min(effective_target_sentence_count, original_sentence_count)

        if effective_target_sentence_count >= original_sentence_count:
            logger.debug(
                f"TextSummarizerNode: Target sentence count ({effective_target_sentence_count}) "
                f"is greater than or equal to original count ({original_sentence_count}). Returning original text."
            )
            return text # No summarization needed, or summarization would make it longer/same.

        summarized_sentences = sentences[:effective_target_sentence_count]
        summarized_text = " ".join(summarized_sentences).strip()

        logger.info(
            f"TextSummarizerNode: Summarized text from {original_sentence_count} sentences "
            f"to {len(summarized_sentences)} sentences (effective ratio: {len(summarized_sentences)/original_sentence_count:.2f})."
        )
        logger.debug(
            f"Original text length: {len(text)}, Summarized text length: {len(summarized_text)}"
        )

        return summarized_text