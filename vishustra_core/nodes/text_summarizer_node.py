import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra node that performs basic text summarization.

    This node takes a string as input and returns a summarized version
    based on a configurable length ratio. It's designed to provide a
    simple, heuristic-based simulation of a text summarization service
    within the orchestration framework.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Summarizes the input text by extracting a fraction of its sentences.

        The summarization logic employs a simple heuristic: it extracts a
        percentage of the original sentences based on the `summary_length_ratio`
        provided in the context. If not specified, it defaults to 30%.
        It ensures at least one sentence is returned if the input is not empty.

        Args:
            data: The input text to be summarized (expected to be a string).
            context: A dictionary containing runtime context and configuration.
                     Expected to contain 'summarizer_config' dictionary, which
                     may hold 'summary_length_ratio' (a float between 0.0 and 1.0).

        Returns:
            A summarized string. Returns an empty string if the input is empty
            or contains only whitespace.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summary_length_ratio' in context is invalid (not
                        between 0.0 exclusive and 1.0 inclusive).
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data).__name__}. "
                         "Expected a string for summarization.")
            raise TypeError("Input data for TextSummarizerNode must be a string.")

        if not data.strip():
            logger.debug("TextSummarizerNode received empty or whitespace-only string, returning as is.")
            return ""

        # Configure summarization parameters from context with a default ratio
        summary_length_ratio = context.get('summarizer_config', {}).get('summary_length_ratio', 0.3)

        if not (0.0 < summary_length_ratio <= 1.0):
            logger.error(f"Invalid 'summary_length_ratio' in context: {summary_length_ratio}. "
                         "Must be between 0.0 (exclusive) and 1.0 (inclusive).")
            raise ValueError("Invalid 'summary_length_ratio' provided for summarization.")

        # Basic sentence splitting (naive, but serves the simulation purpose)
        # Replaces common terminators with '. ' to normalize splitting.
        cleaned_text = data.replace('!', '. ').replace('?', '. ')
        sentences = [s.strip() for s in cleaned_text.split('. ') if s.strip()]

        # Handle cases where no clear sentences are found (e.g., "Hello world")
        if not sentences and data.strip():
            sentences = [data.strip()]

        if not sentences:
            logger.debug("No valid sentences could be extracted, returning original data as fallback.")
            return data

        num_sentences = len(sentences)
        # Calculate number of sentences to keep, ensuring at least one unless text is truly empty
        num_sentences_to_keep = max(1, int(num_sentences * summary_length_ratio))
        # Ensure we don't try to keep more sentences than exist
        num_sentences_to_keep = min(num_sentences_to_keep, num_sentences)

        summarized_sentences = sentences[:num_sentences_to_keep]

        # Re-join sentences and ensure proper termination for readability
        summarized_text = ". ".join(s for s in summarized_sentences if s)
        if summarized_text and not summarized_text.endswith(('.', '!', '?')):
            summarized_text += "."

        logger.info(f"TextSummarizerNode summarized text (original length: {len(data)}, "
                    f"summary length: {len(summarized_text)}) using ratio {summary_length_ratio}.")
        return summarized_text