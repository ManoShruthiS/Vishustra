import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter profanity from text data.

    This node identifies and replaces common profanity words within the input
    text with a masked version (e.g., asterisks), ensuring that the output
    adheres to content guidelines or desired moderation standards.
    """

    # A curated, simple set of profanity words for demonstration purposes.
    # In a production environment, this list would be more extensive,
    # potentially loaded dynamically from a configuration service, database,
    # or an external lexical resource.
    _PROFANE_WORDS = frozenset([
        "ass", "bitch", "bastard", "damn", "fuck", "shit", "cunt",
        "piss", "motherfucker", "cock", "dick", "bollocks", "wanker",
        "fag", "slut"
    ])
    _MASK_CHARACTER = "*"

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out profanity words.

        The method expects the input `data` to be a string. It iterates through
        a predefined list of profanity words and replaces them with a masked
        version (e.g., '****') while preserving the original length of the word.
        The filtering is case-insensitive and performed on whole words only.

        Args:
            data: The input data, expected to be a string containing text
                  to be filtered.
            context: A dictionary providing additional context for the
                     processing operation. This node does not currently
                     utilize context, but it is passed for interface compliance.

        Returns:
            A string identical to the input `data` but with all identified
            profanity words replaced by their masked equivalents.

        Raises:
            TypeError: If the input `data` is not a string, indicating an
                       incorrect data type for this node's operation.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"received '{type(data).__name__}'. Data will not be processed."
            )
            raise TypeError(
                f"{self.node_name} requires string input, "
                f"received {type(data).__name__}"
            )

        processed_text = data
        for word in self._PROFANE_WORDS:
            # Construct a regex pattern for case-insensitive, whole-word matching.
            # re.escape() is used to handle words that might contain regex special characters.
            # \b ensures that only whole words are matched (e.g., 'shit' but not 'shatter').
            pattern = r'\b' + re.escape(word) + r'\b'

            # Create a mask of the same length as the profane word.
            mask = self._MASK_CHARACTER * len(word)

            # Perform case-insensitive replacement using re.sub.
            processed_text = re.sub(pattern, mask, processed_text, flags=re.IGNORECASE)

        logger.debug(
            f"[{self.node_name}] Successfully filtered profanity. "
            f"Original text length: {len(data)}. Processed text length: {len(processed_text)}."
        )
        return processed_text