import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node is available in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanity from text data.
    It supports filtering individual strings or lists of strings.
    Profane words are replaced with asterisks of the same length as the original word.
    """

    # A comprehensive, yet common, set of profanity for filtering.
    # This list can be extended or loaded from a configuration for production use.
    _PROFANITY_WORDS = {
        "anal", "arse", "ass", "bastard", "bitch", "bollocks", "bugger", "bullshit",
        "cunt", "damn", "dick", "douchebag", "fuck", "goddamn", "hell", "jizz",
        "motherfucker", "nigga", "piss", "prick", "pussy", "shit", "slut", "son of a bitch",
        "tits", "wanker", "whore", "crap", "fuck you", "damn it", "mother fucker"
    }

    # Compile regex patterns for faster processing and proper word boundary matching.
    # We use '\b' for word boundaries to avoid censoring parts of non-profane words.
    # re.IGNORECASE makes the filter case-insensitive.
    _PROFANITY_PATTERNS = [
        re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        for word in _PROFANITY_WORDS
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def _filter_text(self, text: str) -> str:
        """
        Internal helper method to filter profanity from a single string.

        Args:
            text: The input string to be filtered.

        Returns:
            The string with profane words replaced by asterisks.
        """
        filtered_text = text
        for pattern in self._PROFANITY_PATTERNS:
            # The replacer function ensures that the replacement string (asterisks)
            # has the same length as the original matched profane word.
            def replacer(match: re.Match) -> str:
                return '*' * len(match.group(0))

            filtered_text = pattern.sub(replacer, filtered_text)
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to remove profanity.

        This method supports two primary data types for filtering:
        1. A single string: The string will be filtered directly.
        2. A list of strings: Each string within the list will be filtered.

        For any other data type, a warning is logged, and the data is returned
        unchanged, maintaining data integrity for unsupported formats.

        Args:
            data: The input data. Expected to be a string or a list of strings.
            context: A dictionary containing contextual information for the current
                     orchestration run (not directly utilized by this node).

        Returns:
            The filtered data (string or list of strings), or the original data
            if its type is not supported for filtering.
        """
        logger.debug(f"[{self.node_name}] Starting profanity filtering process for data type: {type(data)}")

        if isinstance(data, str):
            filtered_data = self._filter_text(data)
            logger.debug(f"[{self.node_name}] Successfully filtered a single string.")
            return filtered_data
        elif isinstance(data, list):
            processed_list = []
            for item_index, item in enumerate(data):
                if isinstance(item, str):
                    processed_list.append(self._filter_text(item))
                else:
                    logger.warning(
                        f"[{self.node_name}] List item at index {item_index} is of type {type(item)}, "
                        "which is not a string. This item will be returned unfiltered."
                    )
                    processed_list.append(item)  # Return non-string items as is
            logger.debug(f"[{self.node_name}] Successfully filtered a list of strings (and preserved non-strings).")
            return processed_list
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported data type for profanity filtering: {type(data)}. "
                "Data will be returned unchanged as filtering cannot be applied."
            )
            return data