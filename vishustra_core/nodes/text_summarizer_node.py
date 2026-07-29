import logging
from typing import Any, Dict
import re

# Assuming this path is correct based on the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that performs basic text summarization.

    This node takes a string as input and generates a summary by selecting
    a configurable percentage of the initial sentences. The summarization
    ratio can be provided via the `context` dictionary.

    If the input data is empty or contains no detectable sentences, it
    handles these cases gracefully.
    """

    _DEFAULT_SUMMARY_RATIO = 0.3  # Default to summarizing to 30% of original sentence count

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to produce a summary.

        The summarization logic primarily involves segmenting the text into
        sentences and then retaining the first `N` sentences, where `N` is
        determined by the `summary_ratio` parameter.

        Args:
            data: The input text (str) that needs to be summarized.
            context: A dictionary containing operational parameters for this node.
                     Expected keys:
                     - 'summary_ratio' (float, optional): A value between 0.0 and 1.0
                       representing the fraction of sentences to retain for the summary.
                       If not provided, the node defaults to `_DEFAULT_SUMMARY_RATIO`.

        Returns:
            str: The summarized text. Returns an empty string if the input
                 data is empty or contains no processable content.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the 'summary_ratio' provided in the context is
                        not a valid float between 0.0 and 1.0.
        """
        logger.debug(f"[{self.node_name}] Starting text summarization process.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected 'str', but received '{type(data).__name__}'.")
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for summarization, "
                f"received type '{type(data).__name__}'."
            )

        stripped_data = data.strip()
        if not stripped_data:
            logger.info(f"[{self.node_name}] Input data is empty or whitespace-only. Returning an empty string.")
            return ""

        # Retrieve summary_ratio from context or use the default value
        summary_ratio = context.get("summary_ratio", self._DEFAULT_SUMMARY_RATIO)

        # Validate summary_ratio
        if not isinstance(summary_ratio, (int, float)) or not (0.0 <= summary_ratio <= 1.0):
            logger.error(f"[{self.node_name}] Invalid 'summary_ratio' in context: '{summary_ratio}'. "
                         "Must be a float between 0.0 and 1.0.")
            raise ValueError(
                f"[{self.node_name}] Invalid 'summary_ratio' in context. "
                f"Expected a float between 0.0 and 1.0, received '{summary_ratio}'."
            )

        # Basic sentence splitting: Splits on periods, exclamation marks, or question marks
        # followed by one or more whitespace characters.
        # The positive lookbehind `(?<=[.!?])` ensures the delimiter is kept with the sentence.
        sentences = re.split(r'(?<=[.!?])\s+', stripped_data)
        # Filter out any empty strings that might result from splitting
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.info(f"[{self.node_name}] No discernible sentences found in the input data. Returning original content.")
            return stripped_data

        # Calculate the number of sentences to keep. Ensure at least one sentence is kept
        # unless summary_ratio is exactly 0 and there are no sentences (handled above).
        num_sentences_to_keep = max(1, int(len(sentences) * summary_ratio))

        # Select the first 'num_sentences_to_keep' sentences
        summarized_sentences = sentences[:num_sentences_to_keep]

        # Join the selected sentences back into a single string.
        # A simple space join works well as the sentence splitting already retains punctuation.
        summarized_text = " ".join(summarized_sentences).strip()

        logger.debug(
            f"[{self.node_name}] Summarization complete. "
            f"Original sentences: {len(sentences)}, Kept: {len(summarized_sentences)}."
        )
        logger.debug(f"[{self.node_name}] Exiting text summarization process.")
        return summarized_text