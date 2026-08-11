import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.
    It condenses input text based on a specified ratio or a default.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a summary.

        Expects `data` to be a string.
        Context can optionally contain `summary_ratio` (float between 0.0 and 1.0)
        to control the length of the summary. If not provided, a default ratio
        of 0.3 (30% of original length) is used.

        Args:
            data: The input text string to be summarized.
            context: A dictionary containing operational context, potentially
                     including 'summary_ratio'.

        Returns:
            A string representing the summarized text.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If 'summary_ratio' in context is not a valid float
                        between 0.0 and 1.0.
        """
        if not isinstance(data, str):
            error_msg = f"Invalid input data type for TextSummarizerNode. Expected str, got {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.strip()
        if not text:
            logger.info("Received empty string for summarization, returning empty string.")
            return ""

        summary_ratio = context.get("summary_ratio")
        default_summary_ratio = 0.3  # Default to 30% summary

        if summary_ratio is None:
            summary_ratio = default_summary_ratio
            logger.debug(f"No 'summary_ratio' provided in context. Using default: {summary_ratio:.2f}.")
        elif not isinstance(summary_ratio, (int, float)) or not (0.0 <= summary_ratio <= 1.0):
            error_msg = (
                f"Invalid 'summary_ratio' in context for TextSummarizerNode. "
                f"Expected a float between 0.0 and 1.0, got {summary_ratio} (type: {type(summary_ratio).__name__})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.debug(f"Using 'summary_ratio' from context: {summary_ratio:.2f}.")

        # A simple simulation: split text into sentences and take a percentage.
        # This is a heuristic approximation for demonstration purposes.
        # Real-world summarization would involve NLP models.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences or (len(sentences) == 1 and not sentences[0]):
            logger.info("Text contains no discernible sentences, returning original text.")
            return text

        num_sentences_to_keep = max(1, int(len(sentences) * summary_ratio))
        
        # Ensure we don't try to get more sentences than available
        summarized_sentences = sentences[:min(num_sentences_to_keep, len(sentences))]
        
        summary = " ".join(summarized_sentences).strip()

        logger.info(
            f"Successfully summarized text (original sentences: {len(sentences)}, "
            f"summary sentences: {len(summarized_sentences)}, ratio: {summary_ratio:.2f})."
        )
        return summary

