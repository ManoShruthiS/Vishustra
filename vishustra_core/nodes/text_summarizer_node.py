import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that performs text summarization.

    This node takes a string as input and produces a shortened version of it,
    simulating a summary. The desired summary length can be configured via the
    context dictionary. It attempts to cut summaries at word boundaries for readability.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a simulated summary.

        Args:
            data (Any): The input data, expected to be a string representing the text to summarize.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Expected to optionally contain 'target_summary_length' (int)
                                      for configuring the approximate summary length. Defaults to 200
                                      characters if not provided or invalid.

        Returns:
            Any: A string representing the simulated summary of the input text.

        Raises:
            ValueError: If the input 'data' is not a string.
            Exception: For any other unexpected errors during the summarization process.
        """
        logger.debug(f"[{self.node_name}] Initiating text summarization process.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type for summarization. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        original_text: str = data
        # Retrieve target_summary_length from context, defaulting to 200
        target_length = context.get("target_summary_length", 200)

        if not isinstance(target_length, int) or target_length <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'target_summary_length' specified in context: {target_length}. "
                "Using default length of 200 characters."
            )
            target_length = 200

        try:
            if len(original_text) <= target_length:
                summary = original_text
                logger.info(
                    f"[{self.node_name}] Original text length ({len(original_text)}) "
                    f"is within or equal to target length ({target_length}). No truncation performed."
                )
            else:
                # Attempt to cut at a word boundary before target_length
                cut_point = original_text.rfind(' ', 0, target_length)
                if cut_point != -1:
                    summary = original_text[:cut_point]
                else:
                    # If no space found (e.g., a very long first word), just hard cut
                    summary = original_text[:target_length]
                summary += "..."
                logger.info(
                    f"[{self.node_name}] Text summarized from {len(original_text)} "
                    f"to approximately {len(summary)} characters."
                )

            logger.debug(f"[{self.node_name}] Text summarization successfully completed.")
            return summary
        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during the summarization process: {e}"
            logger.exception(error_msg) # Logs the full traceback
            raise # Re-raise the exception after logging for upstream handling
