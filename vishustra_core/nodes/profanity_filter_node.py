import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A processing node that filters out common profanities from string data.

    This node identifies and replaces predefined profane words within the input
    text with asterisks. The replacement maintains the original word's length
    for better visual consistency. Profanity detection is case-insensitive.
    """

    _PROFANITIES = {
        "ass", "bitch", "cunt", "damn", "dick", "fuck", "hell", "piss", "shit", "wanker",
        "bollocks", "motherfucker", "bastard"
    }
    """
    A set of profane words to be filtered. This set is defined internally for
    this implementation, but in a production system, it could be loaded from
    a configuration service or dynamically updated.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanities.

        This method expects `data` to be a string. If `data` is not a string,
        a warning is logged, and the original `data` is returned without any
        modifications.

        Profane words are detected using case-insensitive whole-word matching
        and replaced with asterisks, preserving the length of the matched word.

        Args:
            data: The input data, typically expected to be a string containing text.
            context: A dictionary containing contextual information, which is
                     not utilized by this particular node.

        Returns:
            The filtered string if `data` was a string, with profane words
            replaced by asterisks. Otherwise, the original `data` is returned
            unchanged.
        """
        if not isinstance(data, str):
            logger.warning(
                "ProfanityFilterNode received non-string data. "
                "Returning original data without filtering. Received type: %s", type(data)
            )
            return data

        filtered_text = data
        for word in self._PROFANITIES:
            # Construct a regex pattern for case-insensitive whole-word matching.
            # re.escape is used to handle special characters in the profane word.
            # \b ensures that only whole words are matched, preventing partial matches.
            pattern = r'\b' + re.escape(word) + r'\b'

            # Replace all occurrences of the profane word with asterisks.
            # The lambda function ensures the replacement string has the same
            # length as the matched word (m.group(0)).
            filtered_text = re.sub(
                pattern,
                lambda m: '*' * len(m.group(0)),  # m.group(0) is the entire matched string
                filtered_text,
                flags=re.IGNORECASE
            )

        logger.debug("ProfanityFilterNode processed data. Original: '%s', Filtered: '%s'", data, filtered_text)
        return filtered_text
