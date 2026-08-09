import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out specified profanity words from text data.
    It identifies and replaces known profanities (case-insensitively and as whole words)
    with masked characters (asterisks).
    """

    # A simple, static list of profanities for demonstration purposes.
    # In a production Vishustra environment, this list would typically be
    # loaded dynamically from configuration, a database, or an external service
    # for greater flexibility and maintainability.
    _PROFANITY_LIST = [
        "damn",
        "hell",
        "ass",
        "bitch",
        "shit",
        "fuck",
        "crap",
        "piss",
        "cock",
        "cunt"
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        If the input 'data' is a string, this method replaces known profanities
        (case-insensitively and matching whole words) with masked characters
        (e.g., '****' for a four-letter profanity).
        If 'data' is not a string, it logs a warning and returns the data unchanged.

        Args:
            data: The input data, expected to be a string that may contain profanity.
            context: A dictionary containing contextual information relevant to the
                     overall orchestration (not directly used by this node for filtering,
                     but part of the standard `BaseNode` interface).

        Returns:
            The processed data: a string with profanities filtered, or the original
            data if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering will be skipped. Returning original data without modification."
            )
            return data

        processed_text = data
        found_profanity = False

        for word in self._PROFANITY_LIST:
            # Construct a regex pattern for whole word matching, escaping the word
            # to handle potential special regex characters within profanities.
            # Using `\b` for word boundaries ensures "ass" doesn't match "compass".
            pattern = r'\b' + re.escape(word) + r'\b'

            # Check if the profanity exists in the text (case-insensitively)
            if re.search(pattern, processed_text, re.IGNORECASE):
                # If found, replace all occurrences with asterisks of the same length
                replacement = '*' * len(word)
                processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
                found_profanity = True

        if found_profanity:
            logger.info(f"[{self.node_name}] Profanity detected and filtered in input data.")
        else:
            logger.debug(f"[{self.node_name}] No profanity detected in input data. Data unchanged.")

        return processed_text