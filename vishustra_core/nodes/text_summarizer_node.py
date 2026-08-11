import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node responsible for generating a concise summary
    of input text data.

    This node simulates summarization by extracting a configurable number of
    leading sentences from the input, providing a basic but effective
    text condensation mechanism. Future enhancements could integrate
    actual LLM-based summarization models.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this summarizer node.
        """
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, extracting a summary based on the provided
        configuration in the context.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing node-specific
                                       configuration and shared workflow state.
                                       Expected keys:
                                       - 'max_sentences' (int, optional): The maximum
                                         number of leading sentences to include in the summary.
                                         Defaults to 3 if not provided or invalid.

        Raises:
            ValueError: If the input `data` is not a string.

        Returns:
            str: The summarized text, potentially truncated and appended with
                 an ellipsis if the original text was longer.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"received '{type(data).__name__}'."
            )
            raise ValueError(f"Input data for {self.node_name} must be a string.")

        original_text = data.strip()
        if not original_text:
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only text for summarization. Returning empty string.")
            return ""

        original_length = len(original_text)
        logger.debug(f"[{self.node_name}] Initiating summarization for text of length {original_length} characters.")

        # Determine the maximum number of sentences for the summary
        max_sentences = context.get('max_sentences', 3)
        if not isinstance(max_sentences, int) or max_sentences <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'max_sentences' value in context: '{max_sentences}'. "
                "Defaulting to 3 sentences."
            )
            max_sentences = 3

        # A simple regex for sentence tokenization.
        # This works by splitting at common sentence-ending punctuation
        # followed by whitespace, but is not robust for all linguistic nuances.
        sentences = re.split(r'(?<=[.!?])\s+', original_text)

        if not sentences:
            logger.warning(f"[{self.node_name}] No distinct sentences identified in the input text. Returning original text.")
            return original_text # If no sentences, return the original text

        # Select the specified number of leading sentences
        summary_parts = sentences[:max_sentences]
        summarized_text = " ".join(summary_parts)

        # Append an ellipsis if the text was actually truncated
        if len(sentences) > max_sentences:
            summarized_text += "..."

        summarized_length = len(summarized_text)
        logger.info(
            f"[{self.node_name}] Summarization complete. Original length: {original_length} "
            f"chars, Summarized length: {summarized_length} chars. "
            f"Used {min(len(sentences), max_sentences)} of {len(sentences)} sentences."
        )

        return summarized_text