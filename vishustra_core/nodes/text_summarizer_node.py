import logging
from typing import Any, Dict

# Assuming vishustra_core is installed or available in sys.path
# In a real project, this would be relative or part of the package structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    It takes a string as input and returns a truncated version of the text,
    acting as a placeholder for a more sophisticated summarization engine.
    The summary length can be configured via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Simulates summarization of the input text.

        The `data` input is expected to be a string.
        The `context` can optionally contain a 'summary_length' (int)
        parameter to control the target word count of the summary.
        Defaults to 50 words if not specified or invalid.

        Args:
            data: The text content (string) to be summarized.
            context: A dictionary potentially containing processing parameters,
                     e.g., {"summary_length": 100}.

        Returns:
            A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for {self.node_name}. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"{self.node_name} expects string data for summarization. "
                f"Got {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(
                f"{self.node_name} received an empty or whitespace-only string for summarization. "
                "Returning an empty string."
            )
            return ""

        logger.info(f"[{self.node_name}] Initiating summarization for text of length {len(data)} characters.")

        # Determine target word count for the summary
        target_word_count = context.get("summary_length", 50)
        if not isinstance(target_word_count, int) or target_word_count <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid or missing 'summary_length' in context "
                f"(received: {target_word_count}). Defaulting to 50 words."
            )
            target_word_count = 50

        words = data.split()
        summary: str

        if len(words) <= target_word_count:
            summary = data  # Original text is short enough, return as-is
            logger.debug(
                f"[{self.node_name}] Original text word count ({len(words)}) is "
                f"less than or equal to target ({target_word_count}). Returning full text."
            )
        else:
            summary_words = words[:target_word_count]
            summary = " ".join(summary_words) + "..."
            logger.debug(
                f"[{self.node_name}] Summarized text to approximately {target_word_count} words."
            )

        logger.info(f"[{self.node_name}] Summarization complete. Output length: {len(summary)} characters.")
        return summary