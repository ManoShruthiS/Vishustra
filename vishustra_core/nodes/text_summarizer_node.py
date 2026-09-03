import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node that simulates text summarization.

    This node takes a string as input and returns a summarized version
    by extracting the first N sentences. The number of sentences can be
    configured via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input text to generate a summary.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing additional runtime information.
                                      Can include 'summarizer_sentences' (int) to specify
                                      the desired number of sentences in the summary.

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'summarizer_sentences' in context is invalid.
        """
        if not isinstance(data, str):
            logger.error(
                "TextSummarizerNode received non-string data. Expected 'str', but got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"TextSummarizerNode expects string data, but received {type(data).__name__}"
            )

        if not data.strip():
            logger.info("TextSummarizerNode received empty or whitespace-only string, returning it as is.")
            return ""

        # Determine the number of sentences for the summary from context, default to 3.
        # Ensure it's a positive integer.
        num_sentences = context.get("summarizer_sentences", 3)
        if not isinstance(num_sentences, int) or num_sentences <= 0:
            logger.warning(
                "Invalid 'summarizer_sentences' in context: '%s'. "
                "Expected a positive integer. Defaulting to 3 sentences.",
                num_sentences
            )
            num_sentences = 3

        # Simple sentence splitting using regex.
        # A more robust solution might use an NLP library (e.g., NLTK, spaCy).
        sentences = re.split(r'(?<=[.!?])\s+', data.strip())

        if not sentences:
            logger.debug("No sentences found in the input text after splitting.")
            return ""

        if len(sentences) <= num_sentences:
            logger.debug(
                "Input text has %d sentences, which is less than or equal to the "
                "requested summary length of %d sentences. Returning full text.",
                len(sentences), num_sentences
            )
            return data.strip()

        # Join the first 'num_sentences' sentences to form the summary
        summary = " ".join(sentences[:num_sentences])
        logger.info(
            "Successfully summarized text from %d sentences to %d sentences.",
            len(sentences), num_sentences
        )
        return summary