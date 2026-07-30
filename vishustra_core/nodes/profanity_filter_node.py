import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node designed to filter profanity from input text data.

    This node identifies predefined profane words within the input string
    and replaces them with asterisks of the same length. The filtering
    is case-insensitive and ensures whole-word matching to avoid censoring
    parts of legitimate words.
    """

    # A simple, illustrative set of profanities.
    # In a production environment, this list would typically be loaded from
    # a configurable source (e.g., a database, configuration file, or
    # provided via the node's initialization parameters or context).
    _PROFANITIES = {
        "ass", "bitch", "fuck", "shit", "cunt", "damn", "hell", "piss", "bastard",
        "cock", "dick", "fag", "motherfucker", "wanker", "prick", "bollocks",
        "arse", "slut", "whore"
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and filter out profanity.

        If the input `data` is a string, it will be scanned for predefined
        profane words, which will then be replaced by asterisks. If the `data`
        is not a string, a warning is logged, and the original data is returned
        without modification.

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information for the
                     current orchestration run. Not directly used by this node
                     for its core logic but available as part of the node API.

        Returns:
            The processed data with profanity filtered (if `data` was a string),
            or the original `data` if it was not a string or if an unexpected
            error occurred during the filtering process.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Received non-string data of type '%s'. "
                "Profanity filtering will be skipped for this input. Data sample: %.50s",
                self.node_name, type(data).__name__, str(data)
            )
            return data

        filtered_text = data
        original_length = len(data)

        for word in self._PROFANITIES:
            # Construct a regex pattern for case-insensitive, whole-word matching.
            # `re.escape` handles any special regex characters in the profanity word.
            # `\b` asserts a word boundary, ensuring "shit" doesn't match "spreadsheet".
            # `(?i)` makes the match case-insensitive.
            pattern = r'(?i)\b' + re.escape(word) + r'\b'
            replacement = '*' * len(word) # Replace with asterisks of the same length

            try:
                # Use re.sub to replace all occurrences of the pattern
                filtered_text = re.sub(pattern, replacement, filtered_text)
            except Exception as e:
                logger.error(
                    "[%s] An unexpected error occurred while filtering word '%s': %s",
                    self.node_name, word, e
                )
                # Log the error but continue processing to ensure robustness
                # and attempt to filter other words.

        logger.info(
            "[%s] Profanity filtering completed. Original content length: %d, Filtered content length: %d",
            self.node_name, original_length, len(filtered_text)
        )
        return filtered_text
