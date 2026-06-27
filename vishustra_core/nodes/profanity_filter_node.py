import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path for the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node that filters out common profanity from string data.
    It replaces identified profane words with asterisks (***).

    This node is designed to handle string inputs. Non-string inputs will
    be passed through without modification, accompanied by a debug log.
    """

    _PROFANE_WORDS = [
        r'\bfuck\b', r'\bshit\b', r'\bbitch\b', r'\basshole\b',
        r'\bcunt\b', r'\bdamn\b', r'\bhell\b', r'\bnigga\b', r'\bfaggot\b'
    ]
    """
    A list of regular expressions for profane words to be filtered.
    Using word boundaries (`\b`) to prevent partial word matches.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        Args:
            data: The input data, expected to be a string for filtering.
            context: A dictionary containing contextual information
                     (not used for filtering logic in this implementation).

        Returns:
            The processed data, with profanity replaced by '***' if the
            input was a string. Non-string inputs are returned as-is.
        """
        if not isinstance(data, str):
            logger.debug(
                f"[{self.node_name}] Received non-string data of type "
                f"'{type(data).__name__}'. Skipping profanity filtering."
            )
            return data

        filtered_data = data
        original_data_lower = data.lower()
        found_profanity = False

        for profanity_pattern in self._PROFANE_WORDS:
            # Use re.sub for case-insensitive replacement
            # and to handle multiple occurrences
            if re.search(profanity_pattern, original_data_lower):
                filtered_data = re.sub(
                    profanity_pattern,
                    '***',
                    filtered_data,
                    flags=re.IGNORECASE
                )
                found_profanity = True

        if found_profanity:
            logger.info(
                f"[{self.node_name}] Filtered profanity from input string. "
                f"Original (partial): '{data[:50]}...', "
                f"Filtered (partial): '{filtered_data[:50]}...'"
            )
        else:
            logger.debug(
                f"[{self.node_name}] No profanity found in input string. "
                f"Data (partial): '{data[:50]}...'"
            )

        return filtered_data