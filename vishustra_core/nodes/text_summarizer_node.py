import logging
from typing import Any, Dict

# Assuming BaseNode is located at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.

    This node takes a string as input and attempts to produce a shorter
    summary based on parameters provided in the context. It offers basic
    sentence-level summarization.

    Configuration via 'context' dictionary:
    - 'summary_length':
        - If an `int` (e.g., 5), the node will attempt to return
          that many sentences from the beginning of the text.
        - If a `float` between 0.0 and 1.0 (e.g., 0.3), the node
          will attempt to return that percentage of the original sentences.
        - Defaults to 3 sentences if not provided or invalid.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by performing a simulated summarization.

        Args:
            data: The input content to be summarized. Expected to be a string.
            context: A dictionary that may contain 'summary_length' to configure
                     the summarization behavior.

        Returns:
            A string representing the summarized text. Returns an empty string
            if the input data is empty or consists only of whitespace.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input type for TextSummarizerNode. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.warning("Received empty or whitespace-only string for summarization. Returning empty string.")
            return ""

        # Basic sentence splitting (can be enhanced with proper NLP libraries like NLTK or spaCy
        # for more robust sentence boundary detection in a production environment).
        sentences = [s.strip() for s in data.split('.') if s.strip()]

        if not sentences:
            logger.info("No sentences detected in the input text after splitting. Returning empty string.")
            return ""

        target_sentence_count = 3  # Default summary length

        summary_length_param = context.get("summary_length")

        if isinstance(summary_length_param, int) and summary_length_param > 0:
            target_sentence_count = summary_length_param
            logger.debug(f"Context specifies summarization to {target_sentence_count} sentences.")
        elif isinstance(summary_length_param, float) and 0.0 < summary_length_param <= 1.0:
            target_sentence_count = max(1, int(len(sentences) * summary_length_param))
            logger.debug(f"Context specifies summarization to {summary_length_param*100:.0f}% of sentences, resulting in {target_sentence_count} sentences.")
        elif summary_length_param is not None:
            logger.warning(
                f"Invalid 'summary_length' value in context: '{summary_length_param}'. "
                "Expected a positive integer or a float between 0.0 and 1.0. "
                f"Defaulting to {target_sentence_count} sentences."
            )

        # Cap the target count to the actual number of available sentences
        num_sentences_to_summarize = min(target_sentence_count, len(sentences))

        summarized_sentences = sentences[:num_sentences_to_summarize]
        summary_text = ". ".join(summarized_sentences)

        # Ensure the summary ends with a period for grammatical completeness, if not already present
        if summary_text and not summary_text.endswith('.'):
            summary_text += '.'

        logger.info(
            f"Text summarized. Original sentences: {len(sentences)}, "
            f"Summary sentences: {len(summarized_sentences)}."
        )
        return summary_text