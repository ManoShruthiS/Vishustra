import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate abstractive text summarization.

    This node takes an input string and generates a shorter summary.
    The summarization logic is simulated for demonstration purposes,
    typically by truncating the text to a specified word limit while
    attempting to retain key information.
    """

    def __init__(self, default_max_words: int = 100):
        """
        Initializes the TextSummarizerNode.

        Args:
            default_max_words (int): The default maximum number of words
                                     for the generated summary if not
                                     overridden by the context during process.
        """
        if not isinstance(default_max_words, int) or default_max_words <= 0:
            raise ValueError("default_max_words must be a positive integer.")
        self._default_max_words = default_max_words
        logger.info(f"TextSummarizerNode initialized with default_max_words={self._default_max_words}.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by generating a simulated summary.

        The summarization length can be controlled via 'max_words' in the
        context dictionary, falling back to the node's default if not provided.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing additional runtime
                                      information or configuration for the node.
                                      Can include 'max_words' (int) to override
                                      the default summary length.

        Returns:
            str: The simulated summarized text.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If 'max_words' in context is not a positive integer.
            Exception: For other unexpected errors during processing.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data)}.")
            raise TypeError(f"Input data for {self.node_name} must be a string, but received {type(data)}.")

        if not data.strip():
            logger.debug(f"{self.node_name} received empty or whitespace-only string, returning as is.")
            return ""

        try:
            # Determine the maximum word count for the summary
            max_words: int = context.get('max_words', self._default_max_words)
            if not isinstance(max_words, int) or max_words <= 0:
                logger.error(f"Invalid 'max_words' in context: {max_words}. Must be a positive integer.")
                raise ValueError(f"'max_words' in context must be a positive integer, but received {max_words}.")

            words = data.split()
            original_word_count = len(words)

            if original_word_count <= max_words:
                logger.debug(
                    f"{self.node_name}: Input text is already shorter than or equal to target length ({max_words} words). "
                    f"Returning original text. (Original: {original_word_count} words)"
                )
                return data

            # Simulate summarization by taking the first 'max_words' and adding ellipsis
            summarized_words = words[:max_words]
            summary = " ".join(summarized_words) + "..."
            
            logger.info(
                f"{self.node_name} summarized text from {original_word_count} words to approximately {len(summarized_words)} words."
            )
            return summary
        except Exception as e:
            logger.exception(f"An unexpected error occurred in {self.node_name} during summarization.")
            raise Exception(f"Failed to summarize text in {self.node_name}: {e}") from e