import logging
import re
from typing import Any, Dict

# Assuming this path exists in the project structure for Vishustra
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to reduce the length of input text by generating
    a summary. This node supports configurable summarization logic via context
    parameters.

    In its current implementation, this node performs a simple extractive
    summarization by selecting an initial portion of the input text's sentences.
    For a production system, this would typically integrate with an advanced
    Language Model (LLM) or a dedicated text summarization service to perform
    more sophisticated abstractive or extractive summarization.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, expecting a string, and returns a summarized version.

        Args:
            data (Any): The input text to be summarized. Must be a string.
            context (Dict[str, Any]): A dictionary containing processing parameters.
                                      Expected keys (optional):
                                      - 'summary_ratio' (float): The target ratio (0.0 to 1.0)
                                        of the original text's sentences to retain in the summary.
                                        Defaults to 0.3 (30%).
                                      - 'min_sentences' (int): The minimum number of sentences
                                        to include in the summary, regardless of 'summary_ratio'.
                                        Defaults to 2.

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summary_ratio' is outside the valid range [0.0, 1.0]
                        or 'min_sentences' is negative.
        """
        if not isinstance(data, str):
            logger.error(
                f"Input data for TextSummarizerNode must be a string, "
                f"but received {type(data).__name__}. Aborting summarization."
            )
            raise TypeError(
                f"TextSummarizerNode expects string input, but received {type(data).__name__}."
            )

        # Handle empty or whitespace-only input gracefully
        if not data.strip():
            logger.warning(
                "Received empty or whitespace-only text for summarization. Returning an empty string."
            )
            return ""

        # Retrieve and validate context parameters
        summary_ratio = context.get("summary_ratio", 0.3)
        min_sentences = context.get("min_sentences", 2)

        if not (0.0 <= summary_ratio <= 1.0):
            logger.error(
                f"Invalid 'summary_ratio' value: {summary_ratio}. Must be between 0.0 and 1.0."
            )
            raise ValueError("Summary ratio must be between 0.0 and 1.0.")
        if not isinstance(min_sentences, int) or min_sentences < 0:
            logger.error(
                f"Invalid 'min_sentences' value: {min_sentences}. Must be a non-negative integer."
            )
            raise ValueError("Minimum sentences must be a non-negative integer.")

        # Basic sentence tokenization: splits on common sentence-ending punctuation
        # followed by one or more spaces. This is a simple approximation.
        sentences = re.split(r"(?<=[.!?])\s+", data)
        # Filter out any empty strings that might result from tokenization
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.debug(
                "No valid sentences found in the input text after tokenization. "
                "Returning an empty string."
            )
            return ""

        num_original_sentences = len(sentences)
        # Calculate the number of sentences to include based on ratio and minimum
        num_sentences_to_include = max(
            min_sentences, int(num_original_sentences * summary_ratio)
        )

        # Ensure we don't try to include more sentences than are actually available
        num_sentences_to_include = min(
            num_sentences_to_include, num_original_sentences
        )

        # Extractive summarization: take the first N sentences
        summary_sentences = sentences[:num_sentences_to_include]
        summary = " ".join(summary_sentences)

        logger.info(
            f"Summarized text from {num_original_sentences} sentences to "
            f"{len(summary_sentences)} sentences (target ratio: {summary_ratio:.2f}, "
            f"min: {min_sentences})."
        )
        # Log a snippet for debugging, avoiding logging full potentially large texts
        logger.debug(f"Original text start: '{data[:100].replace('\\n', ' ')}...'")
        logger.debug(f"Summarized text start: '{summary[:100].replace('\\n', ' ')}...'")

        return summary