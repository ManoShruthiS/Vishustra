import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates text summarization.

    This node takes a string as input data and produces a summary
    by extracting a portion of the original text. The summarization
    logic is a simple truncation based on a configurable ratio of
    the original text's word count. It's designed to provide a
    placeholder for more advanced summarization models.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to generate a simulated summary.

        Args:
            data: The input text to be summarized. Expected to be a non-empty string.
            context: A dictionary containing execution context and configuration.
                     Can include 'summary_ratio' (float, default 0.3) to control
                     the length of the summary relative to the original text's
                     word count. The ratio must be between 0 (exclusive) and 1 (inclusive).

        Returns:
            A string representing the simulated summary of the input text.
            If the calculated summary length is greater than or equal to the
            original text length, the full original text is returned.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Node '{self.node_name}' received non-string data: {type(data)}. "
                "Expected a string for summarization."
            )
            raise TypeError(
                f"Node '{self.node_name}' expects string data, but received {type(data)}."
            )

        if not data.strip(): # Check for empty or whitespace-only string
            logger.warning(
                f"Node '{self.node_name}' received an empty or whitespace-only string. "
                "Returning an empty string as summary."
            )
            return ""

        summary_ratio = context.get("summary_ratio", 0.3)  # Default to 30% summary
        if not isinstance(summary_ratio, (int, float)) or not (0 < summary_ratio <= 1):
            logger.warning(
                f"Invalid 'summary_ratio' ({summary_ratio}) provided in context for "
                f"Node '{self.node_name}'. Expected a float between (0, 1]. "
                "Falling back to default of 0.3."
            )
            summary_ratio = 0.3

        words = data.split()
        original_word_count = len(words)

        if original_word_count == 0:
            logger.warning(
                f"Node '{self.node_name}' received text with no discernible words "
                "(e.g., only punctuation). Returning empty string."
            )
            return ""

        target_summary_word_count = max(1, int(original_word_count * summary_ratio))

        if target_summary_word_count >= original_word_count:
            logger.info(
                f"Node '{self.node_name}' calculated summary length ({target_summary_word_count} words) "
                f"is greater than or equal to original length ({original_word_count} words). "
                "Returning full original text as summary."
            )
            return data

        summary_words = words[:target_summary_word_count]
        summary = " ".join(summary_words) + "..." # Add ellipsis to indicate truncation

        logger.info(
            f"Node '{self.node_name}' successfully summarized {original_word_count} words "
            f"down to {len(summary_words)} words (ratio: {summary_ratio:.2f})."
        )

        return summary