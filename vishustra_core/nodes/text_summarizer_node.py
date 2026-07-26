import logging
from typing import Any, Dict

# Assuming BaseNode is located in the specified core path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node responsible for generating a summary of input text.

    This node simulates the functionality of an abstractive text summarizer,
    taking a longer piece of text and producing a condensed version.
    The actual summarization logic is a placeholder for demonstration.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to generate a simulated text summary.

        The summarization behavior can be influenced by parameters provided
        in the `context` dictionary.

        Args:
            data: The input data, expected to be a string containing the text
                  to be summarized.
            context: A dictionary containing operational context for the node.
                     - `summary_length_ratio` (float, optional): A value
                       between 0 and 1 indicating the desired summary length
                       as a ratio of the original text's length. Defaults to 0.3.

        Returns:
            A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `summary_length_ratio` in context is invalid.
        """
        logger.debug("TextSummarizerNode received data type: %s", type(data))
        logger.debug("TextSummarizerNode received context: %s", context)

        if not isinstance(data, str):
            logger.error("TextSummarizerNode received non-string data: %s", type(data).__name__)
            raise TypeError(
                f"TextSummarizerNode expects 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.info("TextSummarizerNode received empty or whitespace-only data. Returning empty string.")
            return ""

        original_text = data.strip()
        original_length = len(original_text)
        logger.debug("Original text length: %d characters.", original_length)

        # Validate and retrieve summary_length_ratio from context
        summary_length_ratio = context.get('summary_length_ratio', 0.3)
        if not (isinstance(summary_length_ratio, (int, float)) and 0 < summary_length_ratio <= 1):
            logger.warning(
                "Invalid 'summary_length_ratio' (%s) found in context. "
                "It must be a float or int between 0 and 1. Using default of 0.3.",
                summary_length_ratio
            )
            summary_length_ratio = 0.3

        # Calculate the target character count for the summary.
        # Ensure a minimum useful length for very short texts, and cap at original length.
        target_summary_char_count = max(
            int(original_length * summary_length_ratio),
            min(50, original_length) # Aim for at least 50 chars, but not more than the original text
        )

        simulated_summary: str
        if original_length <= 100:
            # For very short texts, return the text itself as a "summary".
            simulated_summary = original_text
        else:
            # Simulate summarization by truncating the text and appending a marker.
            # This is a basic simulation and not a functional summarization model.
            end_index = min(target_summary_char_count, original_length)
            # Ensure we don't cut off mid-word if possible, though for simulation, simple cut is fine.
            simulated_summary = original_text[:end_index].strip()
            if len(simulated_summary) < original_length:
                simulated_summary += "... [SIMULATED SUMMARY]"

        logger.info(
            "Text summarization simulated. Original length: %d, "
            "Simulated summary length: %d", original_length, len(simulated_summary)
        )
        logger.debug("Simulated summary: %s", simulated_summary)

        return simulated_summary