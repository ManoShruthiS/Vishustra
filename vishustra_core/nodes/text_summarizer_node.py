import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to generate an abstractive summary of input text.

    This node takes a string as input data and produces a shorter, condensed
    version, simulating abstractive summarization logic. Configuration such as
    the target sentence count for the summary can be passed via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to generate a summary.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        to be summarized.
            context (Dict[str, Any]): A dictionary containing runtime context and
                                      configuration parameters.
                                      Expected keys:
                                      - 'target_sentence_count' (int, optional):
                                        The desired number of sentences in the summary.
                                        Defaults to 3.

        Returns:
            Any: A string representing the summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_sentence_count' in context is not a positive integer.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string data: {type(data)}")
            raise TypeError("Input data for TextSummarizerNode must be a string.")

        if not data.strip():
            logger.info("TextSummarizerNode received empty text, returning empty summary.")
            return ""

        target_sentence_count = context.get('target_sentence_count', 3)

        if not isinstance(target_sentence_count, int) or target_sentence_count <= 0:
            logger.error(
                f"Invalid 'target_sentence_count' in context: {target_sentence_count}. "
                "Must be a positive integer."
            )
            raise ValueError(
                "'target_sentence_count' in context must be a positive integer."
            )

        logger.debug(
            f"Summarizing text using target sentence count: {target_sentence_count}"
        )

        # Simulate sentence splitting. A more sophisticated implementation would use
        # NLP libraries (e.g., NLTK), but for simulation, a regex-based split is sufficient.
        sentences = re.split(r'(?<=[.!?])\s+', data.strip())
        
        # Filter out empty strings that might result from splitting
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.info("No meaningful sentences found in the input text, returning original.")
            return data.strip() # Or an empty string, depending on desired behavior for malformed input

        summarized_sentences = sentences[:target_sentence_count]
        summary = " ".join(summarized_sentences)

        # Add an ellipsis if the original text was longer than the summary
        if len(sentences) > len(summarized_sentences):
            summary += "..."
            logger.debug("Added ellipsis as original text was longer.")

        logger.info(
            f"TextSummarizerNode successfully generated summary of length "
            f"{len(summary)} characters."
        )
        return summary

```
