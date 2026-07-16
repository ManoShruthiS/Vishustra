import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

# --- Module-level constants and logger setup ---

_DEFAULT_PROFANITY_LIST: List[str] = [
    "asshole", "bitch", "bastard", "cunt", "damn", "dick", "fuck", "shit",
    "piss", "slut", "whore", "motherfucker", "fucker"
]
"""
A default list of profanity words used if no custom list is provided.
All words are in lowercase for consistent processing.
"""

_DEFAULT_REPLACEMENT_STRING: str = "***"
"""The default string used to replace detected profanity."""

logger = logging.getLogger(__name__)
"""Logger for the ProfanityFilterNode module."""

# --- ProfanityFilterNode Class Definition ---

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out profanity from text data.

    This node identifies and replaces specified profane words within an input string
    with a chosen replacement string (e.g., '***'). It supports both a default
    list of profanity and configurable custom lists, as well as dynamic overrides
    via the processing context.
    """

    def __init__(
        self,
        profanity_list: Optional[List[str]] = None,
        replacement_string: str = _DEFAULT_REPLACEMENT_STRING
    ) -> None:
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list (Optional[List[str]]): An optional custom list of words
                                                  to filter. If None, a default
                                                  list is used. Words will be
                                                  converted to lowercase.
            replacement_string (str): The string to replace detected profanity with.
                                      Defaults to '***'.
        """
        # Normalize profanity list to lowercase and sort by length (descending)
        # to handle cases where one profanity might be a substring of another
        # (e.g., 'shit' and 'bullshit'). Longest words are processed first.
        self._profanity_list: List[str] = sorted(
            [word.lower() for word in (profanity_list or _DEFAULT_PROFANITY_LIST)],
            key=len,
            reverse=True
        )
        self._replacement_string: str = replacement_string
        logger.info(
            f"[{self.node_name}] Initialized with {len(self._profanity_list)} "
            f"profanity words. Replacement string: '{self._replacement_string}'."
        )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        This method expects the `data` to be a string. It iterates through the
        configured profanity list and replaces any occurrences within the text
        with the specified replacement string. The profanity list and replacement
        string can be dynamically overridden for a specific call using the
        `context` dictionary.

        Args:
            data (Any): The input data, which is expected to be a string
                        for filtering.
            context (Dict[str, Any]): A dictionary providing contextual information.
                                      It can contain:
                                      - 'profanity_list' (List[str]): A list of words
                                        to use for filtering, overriding the instance's
                                        default or configured list for this call.
                                      - 'replacement_string' (str): The string to use
                                        for replacement, overriding the instance's
                                        configured string for this call.

        Returns:
            Any: The filtered string if the input was a string and profanity was
                 processed. If the input `data` is not a string, a `TypeError` is raised.

        Raises:
            TypeError: If the input `data` is not a string, as this node is designed
                       to operate on text.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'. Cannot filter non-string data."
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a string for profanity filtering, "
                f"received {type(data).__name__}."
            )

        text_to_filter: str = data

        # Determine the profanity list and replacement string for this specific process call,
        # allowing context overrides.
        current_profanity_list: List[str] = sorted(
            [
                word.lower() for word in
                context.get("profanity_list", self._profanity_list)
            ],
            key=len,
            reverse=True
        )
        current_replacement_string: str = context.get(
            "replacement_string", self._replacement_string
        )

        filtered_text: str = text_to_filter
        for word in current_profanity_list:
            # Create a regex pattern for the word, ensuring whole word matching (\b)
            # and case-insensitive search. re.escape() handles special regex characters.
            pattern = re.compile(
                r'\b' + re.escape(word) + r'\b',
                re.IGNORECASE | re.UNICODE
            )
            filtered_text = pattern.sub(current_replacement_string, filtered_text)
        
        if filtered_text != text_to_filter:
            logger.info(f"[{self.node_name}] Profanity detected and filtered in text.")
        else:
            logger.debug(f"[{self.node_name}] No profanity found in text.")

        return filtered_text