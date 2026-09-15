import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates abstractive text summarization.

    This node takes a string as input and returns a summarized version.
    The summarization is simulated by truncating the text and appending
    an indicator, or by returning the original text if it's already concise.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating text summarization.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Can include 'summary_max_words' (int) to control
                                      the simulated summary length.

        Returns:
            Any: A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Input data for '{self.node_name}' must be a string. Received type: {type(data)}."
            )
            raise TypeError(f"'{self.node_name}' expects string data.")

        text = data.strip()
        if not text:
            logger.info(f"Received empty string for summarization in '{self.node_name}'.")
            return ""

        # Determine target summary length from context, default to 50 words
        # This parameter controls the simulation's output brevity.
        summary_max_words = context.get("summary_max_words", 50)
        if not isinstance(summary_max_words, int) or summary_max_words <= 0:
            logger.warning(
                f"Invalid 'summary_max_words' value in context ({summary_max_words}). "
                "Defaulting to 50 words for '{self.node_name}'."
            )
            summary_max_words = 50

        words = text.split()
        if len(words) <= summary_max_words:
            logger.debug(
                f"Text is already concise ({len(words)} words) for '{self.node_name}', "
                "returning original content."
            )
            return text

        # Simulate abstractive summarization by truncating and adding an indicator
        summarized_words = words[:summary_max_words]
        summary_result = " ".join(summarized_words) + "... (simulated abstractive summary)"

        logger.info(
            f"Successfully simulated summarization for text of {len(words)} words, "
            f"resulting in {len(summarized_words)} words (plus indicator) in '{self.node_name}'."
        )
        return summary_result