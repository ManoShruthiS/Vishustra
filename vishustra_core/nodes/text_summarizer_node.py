import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates text summarization.

    This node takes an input text and produces a condensed version,
    primarily by extracting a specified number of initial words.
    It's designed to simulate the behavior of a summarization component
    within an LLM orchestration framework.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (text) to generate a simulated summary.

        The summarization logic can be configured via the 'context' dictionary.

        Args:
            data (Any): The input data, expected to be a string representing the
                        text to be summarized.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     and configuration parameters.
                                     Expected keys:
                                     - 'summary_word_limit' (int): The maximum
                                       number of words for the summary. Defaults to 50.
                                     - 'summary_prefix' (str): A prefix to add
                                       to the summary. Defaults to "Summary: ".

        Returns:
            Any: A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summary_word_limit' in context is not a positive integer.
        """
        logger.debug(f"[{self.node_name}] Starting text summarization process.")

        # --- Input Validation ---
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"Input data for TextSummarizerNode must be a string, but got {type(data).__name__}")

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only text for summarization. Returning an empty string.")
            return ""

        # --- Configuration Extraction ---
        summary_word_limit = context.get("summary_word_limit", 50)
        summary_prefix = context.get("summary_prefix", "Summary: ")

        if not isinstance(summary_word_limit, int) or summary_word_limit <= 0:
            logger.error(f"[{self.node_name}] Invalid 'summary_word_limit' in context: {summary_word_limit}. Must be a positive integer.")
            raise ValueError(
                f"'summary_word_limit' in context must be a positive integer, "
                f"but got '{summary_word_limit}' (type: {type(summary_word_limit).__name__})"
            )
        
        if not isinstance(summary_prefix, str):
            logger.warning(
                f"[{self.node_name}] 'summary_prefix' in context is not a string. "
                f"Using default prefix 'Summary: ' instead of '{summary_prefix}'."
            )
            summary_prefix = "Summary: " # Fallback to default if type is wrong

        logger.debug(
            f"[{self.node_name}] Configured with word limit: {summary_word_limit}, "
            f"prefix: '{summary_prefix}'"
        )

        # --- Simulated Summarization Logic ---
        words = data.split()
        original_word_count = len(words)

        if original_word_count <= summary_word_limit:
            # If the text is already short enough, return it with the prefix
            summary_text = data
            logger.info(
                f"[{self.node_name}] Text is within word limit ({original_word_count}/{summary_word_limit}). "
                f"Returning original text with prefix."
            )
        else:
            # Otherwise, take the first 'summary_word_limit' words
            summary_words = words[:summary_word_limit]
            summary_text = " ".join(summary_words) + "..."
            logger.info(
                f"[{self.node_name}] Summarized text from {original_word_count} words "
                f"to {len(summary_words)} words."
            )

        final_summary = f"{summary_prefix}{summary_text}"
        logger.debug(f"[{self.node_name}] Summarization complete. Result length: {len(final_summary)} characters.")

        return final_summary
