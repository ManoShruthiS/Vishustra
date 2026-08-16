import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to generate a simulated summary
    of input text.

    This node expects a string as input data and produces a condensed version
    of the text. The summarization process is configurable via the `context`
    dictionary, allowing control over the desired summary length.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to produce a simulated summary.

        The node truncates the input text to a specified number of words,
        acting as a placeholder for more advanced summarization techniques.

        Args:
            data (Any): The input data to be summarized. This node expects
                        a string.
            context (Dict[str, Any]): A dictionary containing operational
                                      parameters for the node.
                                      Expected keys:
                                      - 'summary_length_words' (int, optional):
                                        The maximum number of words the
                                        simulated summary should contain.
                                        If not provided, or if the value is
                                        not a positive integer, it defaults
                                        to 50 words.

        Returns:
            Any: A string representing the simulated summary of the input text.
                 Returns an empty string if the input data is empty or
                 consists only of whitespace.

        Raises:
            TypeError: If the input `data` is not of type string.
        """
        if not isinstance(data, str):
            logger.error(
                f"{self.node_name} received non-string data. "
                f"Expected str, but got {type(data).__name__}."
            )
            raise TypeError(
                f"{self.node_name} expects string input, "
                f"but received {type(data).__name__}"
            )

        stripped_data = data.strip()
        if not stripped_data:
            logger.debug(
                f"{self.node_name} received empty or whitespace-only string, "
                "returning an empty string as summary."
            )
            return ""

        # Determine the target summary length from context
        summary_length_words = context.get('summary_length_words', 50)
        if not isinstance(summary_length_words, int) or summary_length_words <= 0:
            logger.warning(
                f"{self.node_name}: Invalid 'summary_length_words' "
                f"'{summary_length_words}' provided in context. "
                "Defaulting to 50 words for summarization."
            )
            summary_length_words = 50

        words = stripped_data.split()

        if len(words) <= summary_length_words:
            # If the original text is already short enough, return it as is.
            summary = stripped_data
            logger.debug(
                f"{self.node_name}: Input text word count ({len(words)}) "
                "is less than or equal to requested summary length, "
                "returning original text."
            )
        else:
            # Simulate summarization by truncating and adding an ellipsis.
            summary_words = words[:summary_length_words]
            summary = " ".join(summary_words) + "..."
            logger.debug(
                f"{self.node_name}: Successfully generated a simulated summary "
                f"of {len(summary_words)} words."
            )

        return summary