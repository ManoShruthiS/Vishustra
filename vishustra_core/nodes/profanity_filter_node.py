import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanities from text data.

    This node identifies and replaces predefined profane words within an input
    string with a series of asterisks, maintaining the original word length.
    It operates case-insensitively and logs detections.
    """

    # A curated list of regex patterns for common profanities.
    # The '\b' ensures whole word matching.
    # In a production system, this list would likely be externalized,
    # dynamically loaded, or managed by a specialized library.
    _profane_patterns = [
        re.compile(r'\b(?:asshole|ass)\b', re.IGNORECASE),
        re.compile(r'\b(?:bitch|bitches)\b', re.IGNORECASE),
        re.compile(r'\b(?:cunt|cunts)\b', re.IGNORECASE),
        re.compile(r'\b(?:damn|damnit)\b', re.IGNORECASE),
        re.compile(r'\b(?:dick|dicks)\b', re.IGNORECASE),
        re.compile(r'\b(?:fuck|fucker|fucking|fucked)\b', re.IGNORECASE),
        re.compile(r'\b(?:hell)\b', re.IGNORECASE),
        re.compile(r'\b(?:motherfucker)\b', re.IGNORECASE),
        re.compile(r'\b(?:nigger|nigga)\b', re.IGNORECASE),
        re.compile(r'\b(?:shit|shitty)\b', re.IGNORECASE),
        re.compile(r'\b(?:slut|sluts)\b', re.IGNORECASE),
        re.compile(r'\b(?:whore|whores)\b', re.IGNORECASE),
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanities.

        If the input 'data' is a string, it iterates through a predefined list
        of profanity patterns. Any detected profane words are replaced with
        asterisks ('*') of the same length as the original word.
        If 'data' is not a string, a warning is logged, and the data is
        returned unchanged.

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information for the
                     current orchestration run. This node does not explicitly
                     use the context but adheres to the interface.

        Returns:
            The processed data with profanities filtered, or the original data
            if it was not a string.
        """
        logger.debug(f"[{self.node_name}] Initiating profanity filtering process.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data (type: '{type(data).__name__}'). "
                "Profanity filtering is applicable only to strings; data will be returned unchanged."
            )
            return data

        processed_text = data
        detected_profanities = set()

        for pattern in self._profane_patterns:
            matches = list(pattern.finditer(processed_text))
            if matches:
                for match in matches:
                    original_word = match.group(0)
                    detected_profanities.add(original_word.lower())
                    # Replace with asterisks of the same length
                    processed_text = processed_text[:match.start()] + \
                                     '*' * len(original_word) + \
                                     processed_text[match.end():]

        if detected_profanities:
            logger.info(
                f"[{self.node_name}] Detected and filtered profanities: "
                f"{', '.join(sorted(detected_profanities))}."
            )
        else:
            logger.debug(f"[{self.node_name}] No profanities detected in the input text.")

        logger.debug(f"[{self.node_name}] Profanity filtering process completed.")
        return processed_text