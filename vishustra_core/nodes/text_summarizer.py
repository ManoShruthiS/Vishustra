import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra node designed to summarize textual data.

    This node takes a string as input and produces a concise summary.
    The current implementation simulates summarization by extracting
    the initial sentences and/or truncating the text to a specified
    maximum length. This provides a clear interface for future integration
    with advanced NLP models (e.g., transformer-based summarizers).
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating text summarization.

        This method expects `data` to be a string. It applies a simple
        heuristic summarization logic based on sentence extraction and
        character limits, configurable via the `context` dictionary.

        Args:
            data (Any): The input data, expected to be a string of text to summarize.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       or parameters for the node. Supported parameters:
                                       - `max_sentences` (int): Maximum number of sentences
                                                                to include in the summary (default: 3).
                                       - `max_summary_length` (int): Maximum character length
                                                                     of the final summary (default: 250).

        Returns:
            str: A string representing the summarized text.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting text summarization process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type: Expected string, "
                f"got {type(data).__name__}. Returning an empty string."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, "
                f"but received {type(data).__name__}."
            )

        stripped_data = data.strip()
        if not stripped_data:
            logger.warning(
                f"[{self.node_name}] Input text is empty or only whitespace. "
                f"Returning an empty string as summary."
            )
            return ""

        # --- Simulate summarization logic based on context or defaults ---
        max_sentences = context.get('max_sentences', 3)
        max_chars = context.get('max_summary_length', 250)

        logger.info(
            f"[{self.node_name}] Summarizing text with "
            f"max_sentences={max_sentences}, max_chars={max_chars}."
        )

        # Basic sentence tokenization (split by common sentence terminators)
        # Using regex to handle multiple terminators and prevent splitting on decimal points etc.
        sentences = re.split(r'(?<=[.!?])\s+', stripped_data)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.warning(
                f"[{self.node_name}] No valid sentences could be extracted "
                f"from the input. Returning empty string."
            )
            return ""

        # Take up to `max_sentences`
        summary_parts = sentences[:max_sentences]
        simulated_summary = ' '.join(summary_parts)

        # Add an ellipsis if original text had more sentences than included
        if len(sentences) > max_sentences and not simulated_summary.endswith(('...', '.', '!', '?')):
            simulated_summary += "..."

        # Further truncate if it exceeds max_chars
        if len(simulated_summary) > max_chars:
            # Cut at the last whole word before `max_chars` to avoid mid-word truncation
            simulated_summary = simulated_summary[:max_chars].rsplit(' ', 1)[0]
            if not simulated_summary.endswith(('...', '.', '!', '?')):
                simulated_summary += "..."
        
        # Ensure the summary ends with punctuation if it's a full sentence, or ellipsis
        if simulated_summary and not simulated_summary.endswith(('...', '.', '!', '?')):
            simulated_summary += "." # Default to a period if no other terminator

        logger.info(f"[{self.node_name}] Successfully simulated text summarization.")
        logger.debug(
            f"[{self.node_name}] Original text length: {len(data)}, "
            f"Summary length: {len(simulated_summary)}"
        )

        return simulated_summary.strip()