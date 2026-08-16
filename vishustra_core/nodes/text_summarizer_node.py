import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class TextSummarizerNode(BaseNode):
    """
    A Vishustra node designed to extract a concise summary from input text.

    This node simulates text summarization by selecting a subset of sentences
    from the beginning of the input, based on configurable parameters
    provided in the processing context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, expecting a string, and returns a simulated summary.

        The summarization logic primarily involves:
        1. Validating that the input `data` is a string.
        2. Splitting the input text into individual sentences.
        3. Determining the target number of summary sentences based on
           `summary_percentage`, `max_summary_sentences`, and
           `min_summary_sentences` found in the `context`.
        4. Concatenating the selected leading sentences to form the final summary.

        Args:
            data: The input content intended for summarization. Expected to be a string.
            context: A dictionary containing runtime parameters. Relevant keys include:
                     - 'summary_percentage' (float): Desired fraction of original sentences to keep (e.g., 0.25).
                     - 'max_summary_sentences' (int): Maximum number of sentences allowed in the summary.
                     - 'min_summary_sentences' (int): Minimum number of sentences required in the summary.

        Returns:
            A string representing the simulated summary of the input text.

        Raises:
            ValueError: If the input `data` is not a string, or if any unexpected
                        issues arise during the summarization process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to summarize."
            )
            raise ValueError(
                f"TextSummarizerNode requires string input, but received {type(data).__name__}."
            )

        text = data.strip()
        if not text:
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only text for summarization. "
                "Returning an empty string."
            )
            return ""

        logger.info(f"[{self.node_name}] Starting text summarization for content of length {len(text)} characters.")

        try:
            # A basic approach to sentence splitting. For production-grade systems,
            # a more robust NLP library (e.g., NLTK, spaCy) might be preferred.
            sentences = re.split(r'(?<=[.!?])\s+', text)
            total_sentences = len(sentences)

            # Retrieve configuration from context with sensible fallback defaults
            summary_percentage = context.get('summary_percentage', 0.25)
            max_summary_sentences = context.get('max_summary_sentences', total_sentences)
            min_summary_sentences = context.get('min_summary_sentences', 1)

            # Validate and clamp summary_percentage to a reasonable range
            if not (0.0 < summary_percentage <= 1.0):
                logger.warning(
                    f"[{self.node_name}] Invalid 'summary_percentage' ({summary_percentage}) in context. "
                    "Clamping to range (0.0, 1.0]."
                )
                summary_percentage = max(0.01, min(summary_percentage, 1.0))

            # Calculate target sentences based on percentage
            target_sentences_by_percentage = int(total_sentences * summary_percentage)

            # Determine the final number of sentences for the summary, respecting all constraints
            num_summary_sentences = max(
                min_summary_sentences,
                min(target_sentences_by_percentage, max_summary_sentences, total_sentences)
            )

            # Ensure at least one sentence if the original text contained sentences
            if num_summary_sentences == 0 and total_sentences > 0:
                num_summary_sentences = 1
                logger.debug(
                    f"[{self.node_name}] Adjusted summary sentence count to 1 as initial calculation yielded 0, "
                    "but source text has sentences."
                )

            # Limit to actual available sentences if calculated value exceeds them
            if num_summary_sentences > total_sentences:
                num_summary_sentences = total_sentences
                logger.debug(
                    f"[{self.node_name}] Adjusted summary sentence count to {total_sentences} "
                    "as calculated value exceeded total available sentences."
                )
            
            summary_sentences_list = sentences[:num_summary_sentences]
            summary = " ".join(summary_sentences_list).strip()

            logger.info(
                f"[{self.node_name}] Summarization complete. Original text had {total_sentences} sentences. "
                f"Summary generated with {len(summary_sentences_list)} sentences, length: {len(summary)} characters."
            )
            return summary

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during the summarization process.")
            raise ValueError(f"Failed to summarize text due to an internal error: {e}") from e