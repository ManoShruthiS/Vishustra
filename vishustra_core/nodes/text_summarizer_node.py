import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to simulate text summarization.
    
    This node processes an input string, attempting to produce a shorter
    summary string based on a configurable heuristic, primarily by extracting
    leading sentences. It serves as a practical example of text transformation
    within the Vishustra framework, demonstrating input validation, context-aware
    processing, and robust error handling.
    """
    
    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to summarize it.

        The summarization logic is a simple heuristic that extracts a specified
        number of leading sentences from the input text. If the text is shorter
        than the target summary length, the original text is returned.

        Args:
            data: The input text data, which must be a string.
            context: A dictionary containing additional runtime context.
                     Supports 'summary_sentence_count' (int) to specify the
                     number of sentences for the summary. Defaults to 3.

        Returns:
            A summarized version of the input text (str). An ellipsis "..." is
            appended if truncation occurs. Returns the original text if it's
            already short or if the input is an empty string.

        Raises:
            ValueError: If the input 'data' is not a string, indicating an
                        invalid input type for this processing node.
        """
        if not isinstance(data, str):
            logger.error(
                f"TextSummarizerNode received non-string data. Expected 'str', got '{type(data).__name__}'."
            )
            raise ValueError("Input data for TextSummarizerNode must be a string.")

        clean_data = data.strip()
        if not clean_data:
            logger.info("TextSummarizerNode received an empty string after stripping whitespace. Returning as is.")
            return ""

        # Determine target summary sentence count from context or use a default
        target_summary_sentences = 3  # Default value
        if 'summary_sentence_count' in context:
            try:
                requested_count = int(context['summary_sentence_count'])
                if requested_count > 0:
                    target_summary_sentences = requested_count
                else:
                    logger.warning(
                        f"Invalid 'summary_sentence_count' in context: {requested_count}. "
                        "Must be a positive integer. Using default value: {target_summary_sentences}."
                    )
            except (TypeError, ValueError):
                logger.warning(
                    f"Non-integer or invalid 'summary_sentence_count' in context: "
                    f"'{context['summary_sentence_count']}'. Using default value: {target_summary_sentences}."
                )

        # Split the input text into sentences.
        # This regex splits by common sentence terminators (. ! ?) followed by optional whitespace.
        # We also filter out empty strings that might result from splitting.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s*', clean_data) if s.strip()]
        
        num_sentences = len(sentences)
        
        if num_sentences <= target_summary_sentences:
            logger.debug(
                f"Text has {num_sentences} sentences, which is less than or equal to the "
                f"target of {target_summary_sentences}. Returning original text without summarization."
            )
            return clean_data
        
        # Construct the summary from the leading sentences
        summary_parts = sentences[:target_summary_sentences]
        summary = " ".join(summary_parts)
        
        # Append an ellipsis to clearly indicate that the text has been truncated
        summary += "..."
            
        logger.info(
            f"Text summarized from {num_sentences} sentences down to {len(summary_parts)} leading sentences."
        )
        return summary