import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to generate a simulated summary of input text.

    This node truncates the input text based on a specified maximum word count,
    which can be configured via the processing context. It's intended to
    simulate a basic summarization operation within a larger workflow.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, attempting to summarize it by truncation.

        The `data` input is expected to be a string. If it's not, a TypeError
        is raised. If the string is empty or only whitespace, an empty string
        is returned.

        Summarization length is controlled by `context['summary_max_words']`.
        If present and a positive integer, the text will be truncated to this
        word count. Otherwise, a default of 50 words is used. If the original
        text is shorter than the target `summary_max_words`, the full text
        is returned.

        Args:
            data: The input text (str) that needs to be summarized.
            context: A dictionary potentially containing configuration for
                     summarization, e.g., 'summary_max_words' (int).

        Returns:
            A string representing the simulated summary of the input text.
            An ellipsis "..." is appended if the text was truncated.

        Raises:
            TypeError: If `data` is not a string.
            ValueError: If `summary_max_words` in context is not a positive integer.
        """
        if not isinstance(data, str):
            logger.error(
                "TextSummarizerNode received non-string data. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"Input data for TextSummarizerNode must be a string, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.info(
                "TextSummarizerNode received an empty or whitespace-only string. Returning empty string."
            )
            return ""

        summary_max_words: Any = context.get("summary_max_words", 50)

        if not isinstance(summary_max_words, int) or summary_max_words <= 0:
            logger.error(
                "Invalid 'summary_max_words' value in context: '%s'. Must be a positive integer.",
                summary_max_words
            )
            raise ValueError(
                f"'summary_max_words' in context must be a positive integer, "
                f"but received {summary_max_words}."
            )

        logger.debug(
            "TextSummarizerNode initiated processing with 'summary_max_words': %d.",
            summary_max_words
        )

        words = data.split()

        if len(words) <= summary_max_words:
            logger.debug(
                "Original text word count (%d) is less than or equal to 'summary_max_words' (%d). "
                "Returning full text.",
                len(words), summary_max_words
            )
            return data
        else:
            summary = " ".join(words[:summary_max_words])
            logger.info(
                "Text summarized by truncating to %d words (original %d words).",
                summary_max_words, len(words)
            )
            return summary + "..."