import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    This node takes a string as input and returns a summarized version.
    The summarization logic is a simple extraction of the first N sentences,
    where N can be configured via the 'summary_length' key in the context.
    This serves as a placeholder for more sophisticated NLP-based summarization
    models in a full orchestration.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Summarizes the input text data.

        This method extracts the first `summary_length` sentences from the input text.
        The `summary_length` can be specified in the `context` dictionary;
        otherwise, it defaults to 3 sentences.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Can include 'summary_length' (int) to specify
                                      the maximum number of sentences for the summary.

        Returns:
            str: The summarized text.

        Raises:
            ValueError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'. Data: {data!r}"
            )
            raise ValueError(f"Input data for {self.node_name} must be a string.")

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only text for summarization.")
            return ""

        summary_length = context.get("summary_length", 3)
        if not isinstance(summary_length, int) or summary_length <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'summary_length' value '{summary_length}' in context. "
                "Expected a positive integer. Defaulting to 3 sentences for summarization."
            )
            summary_length = 3

        # A basic sentence tokenization using regex. This pattern splits by common
        # sentence-ending punctuation (. ! ?) followed by any whitespace.
        # The positive lookbehind `(?<=[.!?])` ensures the punctuation is included
        # in the preceding sentence part.
        sentences = re.split(r'(?<=[.!?])\s*', data)
        
        # Filter out any empty strings that might result from the split (e.g., if text ends with two periods)
        # and strip leading/trailing whitespace from each sentence.
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.warning(f"[{self.node_name}] Could not extract any discernible sentences from the input text.")
            return ""

        # Take the first N sentences as the summary.
        summarized_sentences = sentences[:summary_length]
        
        # Rejoin the selected sentences with a space.
        summary = " ".join(summarized_sentences)
        
        # Ensure the summary ends with appropriate punctuation for readability, if not already present.
        if summary and not re.search(r'[.!?]$', summary):
            summary += '.'

        logger.info(
            f"[{self.node_name}] Successfully summarized text to {len(summarized_sentences)} sentence(s)."
        )
        return summary