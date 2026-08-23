import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra node that performs text summarization.

    This node takes a string as input and generates a summarized version
    based on configurable parameters provided in the context. It simulates
    an abstractive summarization process by intelligently truncating text
    to a specified maximum length, aiming to preserve word boundaries.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to generate a text summary.

        Args:
            data (Any): The input data, expected to be a string representing the text to summarize.
            context (Dict[str, Any]): A dictionary containing parameters for summarization.
                                       Expected keys:
                                       - 'max_summary_length' (int, optional): The maximum
                                         character length of the generated summary. Defaults to 250.
                                         Must be a positive integer.
                                       - 'summarization_method' (str, optional): Placeholder for
                                         future integration with different summarization models
                                         (e.g., 'abstractive', 'extractive'). Not actively used
                                         in this simulated version but can be passed.

        Returns:
            Any: A string representing the summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'max_summary_length' in context is not a positive integer.
        """
        logger.debug(f"[{self.node_name}] Starting process for input data (type: {type(data).__name__}).")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'. "
                f"Cannot summarize non-string data."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        input_text: str = data
        
        # Retrieve and validate max_summary_length from context
        max_summary_length: int = context.get('max_summary_length', 250)
        
        if not isinstance(max_summary_length, int) or max_summary_length <= 0:
            error_msg = (
                f"[{self.node_name}] Invalid 'max_summary_length' parameter in context. "
                f"Expected a positive integer, but received '{max_summary_length}' "
                f"(type: {type(max_summary_length).__name__})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"[{self.node_name}] Summarizing text of {len(input_text)} characters "
            f"to a maximum of {max_summary_length} characters."
        )

        # Simulate abstractive summarization via intelligent truncation
        if len(input_text) <= max_summary_length:
            summarized_text = input_text
            logger.debug(f"[{self.node_name}] Input text length ({len(input_text)}) is within "
                         f"max_summary_length ({max_summary_length}). No truncation performed.")
        else:
            # Attempt to truncate at the last word boundary before max_summary_length
            # to avoid cutting words in half.
            truncated_segment = input_text[:max_summary_length]
            last_space_index = truncated_segment.rfind(' ')

            if last_space_index != -1 and last_space_index > max_summary_length * 0.75:
                # Only use the last space if it's not too far back,
                # preventing very short summaries if the first word is extremely long.
                summarized_text = truncated_segment[:last_space_index] + "..."
            else:
                # Fallback to direct truncation if no suitable space is found
                summarized_text = truncated_segment + "..."
            logger.debug(f"[{self.node_name}] Text truncated from {len(input_text)} to {len(summarized_text)} characters.")

        logger.info(f"[{self.node_name}] Text summarization completed. Output length: {len(summarized_text)} characters.")
        return summarized_text
