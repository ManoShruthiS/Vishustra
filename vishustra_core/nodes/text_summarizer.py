import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizer(BaseNode):
    """
    A Vishustra processing node that simulates text summarization.

    This node takes a string as input and produces a summarized version.
    For demonstration purposes, summarization is simulated via truncation
    to a configurable word count. In a production environment, this would
    interface with a sophisticated summarization model or an LLM.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a summary.

        Args:
            data (Any): The input data expected to be a string.
            context (Dict[str, Any]): A dictionary containing runtime
                                      information and configurable parameters.
                                      Expected keys include:
                                      - "target_word_count" (int, optional):
                                        The desired maximum word count for the summary.
                                        Defaults to 50 if not provided.

        Returns:
            Any: The summarized text as a string.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error("TextSummarizer received non-string input. Type: %s", type(data).__name__)
            raise TypeError(
                f"TextSummarizer expects 'data' to be a string, "
                f"but received type: {type(data).__name__}"
            )

        if not data.strip():
            logger.warning("TextSummarizer received an empty or whitespace-only string. Returning empty string.")
            return ""

        # Retrieve summarization parameters from context with a default fallback
        target_word_count = context.get("target_word_count", 50)

        if not isinstance(target_word_count, int) or target_word_count <= 0:
            logger.warning(
                "Invalid 'target_word_count' (%s) in context. "
                "Defaulting to 50 words.",
                target_word_count
            )
            target_word_count = 50

        # Simulate summarization by truncating the text to the target word count.
        # In a real scenario, this would involve calling a summarization model (e.g., LLM API).
        words = data.split()

        if len(words) <= target_word_count:
            summary = data
            logger.debug("Input text length (%d words) is within or below target word count (%d). Returning original text.", len(words), target_word_count)
        else:
            summary_words = words[:target_word_count]
            summary = " ".join(summary_words) + "..."
            logger.debug("Summarized text to approximately %d words from %d original words.", target_word_count, len(words))
            
        return summary