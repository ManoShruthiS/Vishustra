import logging
import re
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates abstractive text summarization.

    This node takes a string as input and produces a summarized version
    by selecting a specified percentage of the original sentences. It includes
    parameters for controlling the summary length and handling short inputs
    gracefully.
    """

    def __init__(self, summary_ratio: float = 0.3, min_sentences: int = 1):
        """
        Initializes the TextSummarizerNode.

        Args:
            summary_ratio (float): The target ratio of sentences to retain from the
                                   original text (e.g., 0.3 means aim for 30% of sentences).
                                   Must be a value between 0 (exclusive) and 1 (inclusive).
            min_sentences (int): The minimum number of sentences to include in the summary.
                                 This ensures that even very short texts or those where
                                 `summary_ratio` yields few sentences will produce at
                                 least this many sentences, up to the total available.
                                 Must be a non-negative integer.
        Raises:
            ValueError: If `summary_ratio` or `min_sentences` are out of their valid ranges.
        """
        if not (0 < summary_ratio <= 1):
            raise ValueError("summary_ratio must be between 0 (exclusive) and 1 (inclusive).")
        if not (min_sentences >= 0):
            raise ValueError("min_sentences must be a non-negative integer.")

        self._summary_ratio = summary_ratio
        self._min_sentences = min_sentences
        logger.debug(
            f"[{self.node_name}] Initialized with summary_ratio={self._summary_ratio}, "
            f"min_sentences={self._min_sentences}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by simulating text summarization.

        It segments the input text into sentences and selects a subset based on
        the configured `summary_ratio` and `min_sentences` parameters.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       (e.g., configuration, runtime parameters). Not directly
                                       used for summarization logic in this node, but available
                                       for future extensions.

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input data is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting process for data of type: {type(data)}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"[{self.node_name}] Input 'data' must be a string, but got {type(data).__name__}.")

        stripped_data = data.strip()
        if not stripped_data:
            logger.warning(f"[{self.node_name}] Input text is empty or only whitespace. Returning empty string.")
            return ""

        # Simple sentence tokenization using regex. Handles common punctuation.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', stripped_data) if s.strip()]

        if not sentences:
            logger.warning(
                f"[{self.node_name}] No discernable sentences found in input. "
                "Returning original (cleaned) text."
            )
            return stripped_data

        num_original_sentences = len(sentences)

        # Calculate target number of sentences based on ratio and minimum
        target_sentences_by_ratio = int(num_original_sentences * self._summary_ratio)
        target_summary_sentences = max(self._min_sentences, target_sentences_by_ratio)
        
        # Ensure we don't try to take more sentences than available in the original text
        sentences_to_take = min(target_summary_sentences, num_original_sentences)

        # If min_sentences is 0 and ratio also leads to 0, but text exists, take at least one if possible
        if sentences_to_take == 0 and num_original_sentences > 0:
            sentences_to_take = 1 # Fallback to 1 sentence if nothing else applies
            logger.debug(
                f"[{self.node_name}] Calculated sentences to take was 0, but original text has content. "
                "Defaulting to 1 sentence."
            )

        if sentences_to_take >= num_original_sentences:
            logger.info(
                f"[{self.node_name}] Summary length ({sentences_to_take} sentences) "
                f"is greater than or equal to original length ({num_original_sentences} sentences). "
                "Returning original text."
            )
            return stripped_data # Effectively returning the original if no real summarization happens

        # Simulate summarization by taking the first 'sentences_to_take' sentences
        summarized_sentences = sentences[:sentences_to_take]
        summarized_text = " ".join(summarized_sentences)

        logger.info(
            f"[{self.node_name}] Successfully summarized text from {num_original_sentences} "
            f"sentences to {len(summarized_sentences)} sentences (effective ratio: "
            f"{len(summarized_sentences)/num_original_sentences:.2f})."
        )
        logger.debug(f"[{self.node_name}] Process completed.")
        return summarized_text

