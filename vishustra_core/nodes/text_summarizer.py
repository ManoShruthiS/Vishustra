import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node responsible for generating a summary of
    input text. It simulates summarization by extracting a subset of
    sentences based on configuration provided in the context or default heuristics.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a string, and returns a summarized version.

        The summarization logic can be controlled via the 'context' dictionary:
        - 'summary_length_ratio' (float): Desired length of the summary as a
                                          ratio (0.0 to 1.0) of the original text's
                                          sentence count. Takes precedence over
                                          'max_sentences'.
        - 'max_sentences' (int): Maximum number of sentences to include in the summary.
                                 Defaults to 3 if no ratio is provided.

        Args:
            data: The input text as a string to be summarized.
            context: A dictionary containing operational parameters for summarization.

        Returns:
            A string representing the summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summary_length_ratio' or 'max_sentences' in context are invalid.
        """
        if not isinstance(data, str):
            logger.error(
                "TextSummarizerNode received non-string data. Expected type 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError("Input data for TextSummarizerNode must be a string.")

        text = data.strip()
        if not text:
            logger.warning("TextSummarizerNode received empty or whitespace-only text. Returning empty string.")
            return ""

        # A basic sentence splitting for demonstration.
        # In a production system, a more robust NLP library (e.g., NLTK, spaCy)
        # would typically be used for precise sentence tokenization.
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.warning(
                "TextSummarizerNode could not split text into discernible sentences. Returning original text."
            )
            return text

        num_sentences_to_summarize: int = 0
        total_sentences = len(sentences)

        # Prioritize summary length ratio if provided and valid
        summary_length_ratio = context.get('summary_length_ratio')
        if isinstance(summary_length_ratio, (int, float)):
            if not (0.0 < summary_length_ratio <= 1.0):
                logger.error(
                    "Invalid 'summary_length_ratio' in context: %s. Must be between 0.0 and 1.0 (exclusive of 0).",
                    summary_length_ratio
                )
                raise ValueError("Invalid 'summary_length_ratio'. Must be between 0.0 and 1.0.")
            num_sentences_to_summarize = max(1, int(total_sentences * summary_length_ratio))
            logger.debug(
                "Summarizing with ratio %.2f, targeting %d of %d sentences.",
                summary_length_ratio, num_sentences_to_summarize, total_sentences
            )
        else:
            # Fallback to max_sentences parameter
            max_sentences = context.get('max_sentences', 3)  # Default to 3 sentences
            if not isinstance(max_sentences, int) or max_sentences <= 0:
                logger.error(
                    "Invalid 'max_sentences' in context: %s. Must be a positive integer.",
                    max_sentences
                )
                raise ValueError("Invalid 'max_sentences'. Must be a positive integer.")
            num_sentences_to_summarize = min(total_sentences, max_sentences)
            logger.debug(
                "Summarizing with max sentences %d, targeting %d of %d sentences.",
                max_sentences, num_sentences_to_summarize, total_sentences
            )

        # Ensure we always attempt to return at least one sentence if available
        if num_sentences_to_summarize == 0 and total_sentences > 0:
            logger.warning(
                "Calculated 0 sentences for summary, but text contains sentences. Defaulting to 1 sentence."
            )
            num_sentences_to_summarize = 1
        elif num_sentences_to_summarize == 0: # Case where total_sentences is also 0
             return ""


        summary_sentences = sentences[:num_sentences_to_summarize]
        summary = " ".join(summary_sentences)

        logger.info(
            "Successfully summarized text (original length: %d chars, summary length: %d chars).",
            len(text), len(summary)
        )
        return summary