from vishustra_core.nodes.base_node import BaseNode
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out profanity from text data.
    It replaces detected profane words with a configurable masked string,
    enhancing the safety and appropriateness of generated content.
    """

    # A simple, illustrative set of profane words for demonstration.
    # In a production environment, this list would typically be externalized
    # (e.g., loaded from a configuration file, a database, or integrated with
    # a dedicated profanity detection library) and potentially more extensive.
    _PROFANE_WORDS = {
        "badword", "damn", "hell", "shit", "fuck", "asshole", "bitch", "cunt"
    }
    _REPLACEMENT_MASK = "***"

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and filter out profanity.

        If the input `data` is a string, the method performs a case-insensitive
        replacement of known profane words with a defined mask (`***`).
        If `data` is not a string, a warning is logged, and the data is
        returned without modification.

        Args:
            data: The input data, expected to be a string for profanity filtering.
                  Non-string data will be passed through untouched.
            context: A dictionary containing contextual information relevant
                     to the node's operation (e.g., global settings, session data).
                     Currently, this node does not utilize context for its core logic.

        Returns:
            The processed data with profanity filtered (if `data` was a string),
            or the original data otherwise.
        """
        logger.debug(f"ProfanityFilterNode received data for processing. Type: {type(data)}")

        if not isinstance(data, str):
            logger.warning(
                f"ProfanityFilterNode received non-string data (type: {type(data)}). "
                "Skipping profanity filtering and returning data unchanged."
            )
            return data

        processed_text = data
        profanity_detected = False

        for word in self._PROFANE_WORDS:
            # We use a helper for case-insensitive replacement to ensure broad coverage.
            # re.escape is used to handle potential special regex characters within
            # the profane words themselves, making the pattern safe.
            # The word boundary (\b) ensures we only match whole words,
            # preventing "scunthorpe problem" type issues.
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, processed_text, re.IGNORECASE):
                profanity_detected = True
                processed_text = re.sub(pattern, self._REPLACEMENT_MASK, processed_text, flags=re.IGNORECASE)
                logger.debug(f"Replaced instances of '{word}' in text.")
        
        if profanity_detected:
            logger.info("Profanity detected and filtered from text.")
        else:
            logger.debug("No profanity detected in text.")

        return processed_text

# Note: The BaseNode class provided in the problem description is assumed
# to be correctly imported from 'vishustra_core.nodes.base_node'.
