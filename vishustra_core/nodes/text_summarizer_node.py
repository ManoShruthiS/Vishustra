import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

# Initialize a logger for this module.
logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates text summarization.

    This node truncates input text to a specified maximum length, attempting
    to end on a word boundary and appending an ellipsis to indicate truncation.
    """

    def __init__(self, max_summary_length: int = 200):
        """
        Initializes the TextSummarizerNode.

        Args:
            max_summary_length (int): The maximum desired length for the summary,
                                      including space for the ellipsis if truncation occurs.
                                      Must be a positive integer.
        """
        if not isinstance(max_summary_length, int) or max_summary_length <= 0:
            logger.error(f"Invalid max_summary_length provided: {max_summary_length}. It must be a positive integer.")
            raise ValueError("max_summary_length must be a positive integer.")
            
        self.max_summary_length = max_summary_length
        logger.debug(f"Initialized TextSummarizerNode with max_summary_length={self.max_summary_length}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by generating a simulated summary.

        The summary is created by truncating the input text to `max_summary_length`,
        attempting to end on a word boundary and appending an ellipsis if truncation occurs.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. (Not directly
                                       used by this node but passed as per BaseNode contract).

        Returns:
            Any: The summarized text (string).

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        # --- Input Validation ---
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data)}. Expected string.")
            raise TypeError("TextSummarizerNode expects string input for summarization.")

        cleaned_data = data.strip()
        if not cleaned_data:
            logger.warning("TextSummarizerNode received an empty or whitespace-only string. Returning an empty string.")
            return ""

        # Log the beginning of processing, showing a snippet of the input data.
        log_snippet = cleaned_data[:70] + ('...' if len(cleaned_data) > 70 else '')
        logger.info(f"Processing text for summarization (snippet): '{log_snippet}'")

        # --- Summarization Logic (Simulation) ---
        # If the text is shorter than or equal to the maximum length, return it as is.
        if len(cleaned_data) <= self.max_summary_length:
            logger.debug(f"Input text length ({len(cleaned_data)}) is <= max_summary_length ({self.max_summary_length}). Returning full text.")
            return cleaned_data

        # Define the ellipsis string and its length.
        ellipsis = "..."
        ellipsis_len = len(ellipsis)

        # Calculate the effective length for the content before adding the ellipsis.
        # Ensure there's enough space for the ellipsis itself.
        if self.max_summary_length <= ellipsis_len:
            logger.warning(
                f"Configured max_summary_length ({self.max_summary_length}) is too small to add a meaningful "
                f"ellipsis. Truncating text directly to {self.max_summary_length} characters."
            )
            return cleaned_data[:self.max_summary_length].strip()

        effective_content_length = self.max_summary_length - ellipsis_len
        
        # Take the initial segment of the text up to the effective content length.
        summary_segment = cleaned_data[:effective_content_length]
        
        # Attempt to find a word boundary to make the truncation less abrupt.
        # We search for the last space character within the summary_segment.
        last_space_index = summary_segment.rfind(' ')

        final_summary: str
        # If a space is found relatively close to the end of the segment,
        # we cut there and append the ellipsis. This heuristic aims for better readability.
        if last_space_index != -1 and last_space_index > effective_content_length * 0.7: 
            final_summary = summary_segment[:last_space_index].strip() + ellipsis
        else:
            # If no suitable word boundary is found, or if the segment is mostly one long word,
            # we just append the ellipsis to the hard-cut segment.
            final_summary = summary_segment.strip() + ellipsis
            
        logger.debug(f"Generated summary (length {len(final_summary)}): '{final_summary}'")
        return final_summary

