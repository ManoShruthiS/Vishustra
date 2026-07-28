import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is located in this path relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizer(BaseNode):
    """
    A processing node that simulates text summarization by truncating
    the input text to a specified number of words.

    It expects a string as input data and can take an optional 'summary_length'
    parameter from the context dictionary to control the output length.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to produce a simulated summary.

        The summary length can be controlled via the 'summary_length' key
        in the context dictionary. If not provided or invalid, a default
        length of 50 words will be used.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing additional runtime information.
                                       Expected to optionally contain 'summary_length' (int).

        Returns:
            Any: A string representing the summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If an unexpected error occurs during summarization.
        """
        logger.debug(f"[{self.node_name}] Starting text summarization process.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        original_text: str = data
        words = original_text.split()
        original_word_count = len(words)

        # Retrieve summary_length from context, defaulting to 50 words if not provided
        summary_length: int = context.get("summary_length", 50)

        if not isinstance(summary_length, int) or summary_length <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid or non-positive 'summary_length' "
                f"({summary_length}) found in context. Falling back to default of 50 words."
            )
            summary_length = 50

        logger.info(
            f"[{self.node_name}] Attempting to summarize text (original words: {original_word_count}) "
            f"to approximately {summary_length} words."
        )

        if original_word_count <= summary_length:
            logger.info(
                f"[{self.node_name}] Original text is shorter than or equal to "
                f"the requested summary length. Returning original text."
            )
            return original_text

        try:
            # Simulate summarization by taking the first 'summary_length' words
            summarized_words = words[:summary_length]
            summarized_text = " ".join(summarized_words) + "..."
            
            logger.debug(
                f"[{self.node_name}] Summarization complete. "
                f"Output word count: {len(summarized_words)}."
            )
            return summarized_text
        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during summarization: {e}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from e
