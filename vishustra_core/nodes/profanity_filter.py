import logging
import re
from typing import Any, Dict, List, Optional

# Assuming this path exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanity from text data.

    This node replaces specified profane words with a masked sequence (e.g., '***').
    It uses regular expressions with word boundaries for accurate detection and
    offers configurable profanity lists, replacement strings, and case sensitivity.
    It is designed to work primarily with string inputs, logging a warning and returning
    the original data if a non-string type is encountered.
    """

    # A default list of common profanities. This can be extended or externalized
    # in future iterations via configuration mechanisms.
    _default_profanity_list = [
        "fuck", "shit", "asshole", "bitch", "cunt", "damn", "hell", "piss",
        "wanker", "motherfucker", "bastard", "tits", "cock", "dick", "pussy",
        "bollocks", "arse", "slag", "prick"
    ]
    _default_replacement_string = "***"

    def __init__(
        self,
        profanity_list: Optional[List[str]] = None,
        replacement_string: Optional[str] = None,
        case_sensitive: bool = False
    ):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list: An optional list of custom profane words to filter.
                            If None, a default list is used. Words in this list
                            will be treated based on `case_sensitive`.
            replacement_string: An optional string to replace profanities with.
                                If None, '***' is used.
            case_sensitive: If True, the filter will be case-sensitive.
                            Defaults to False, meaning 'fUcK' will be filtered
                            if 'fuck' is in the list.
        """
        self._profanity_list = profanity_list if profanity_list is not None else self._default_profanity_list
        self._replacement_string = replacement_string if replacement_string is not None else self._default_replacement_string
        self._case_sensitive = case_sensitive

        # Compile regex patterns for efficient and case-sensitive replacement
        # Using word boundaries (\b) to avoid replacing parts of non-profane words
        # e.g., 'scunthorpe' should not become 's***horpe'.
        # re.escape() is used to handle special regex characters that might be
        # present in the profane words themselves.
        self._compiled_patterns = []
        flags = 0 if self._case_sensitive else re.IGNORECASE
        for word in self._profanity_list:
            self._compiled_patterns.append(re.compile(r'\b' + re.escape(word) + r'\b', flags))

        logger.debug(
            f"[{self.node_name}] Node initialized with "
            f"profanity_list_size={len(self._profanity_list)}, "
            f"replacement='{self._replacement_string}', case_sensitive={case_sensitive}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        If the input `data` is a string, it iterates through the compiled
        profanity patterns and replaces occurrences of defined profane words
        with the configured replacement string. If `data` is not a string,
        it logs a warning and returns the original data unchanged, adhering
        to a robust processing pipeline design.

        Args:
            data: The input data, expected to be a string for filtering.
            context: A dictionary containing contextual information (currently not used by this node,
                     but passed for compliance with BaseNode interface).

        Returns:
            The processed data (string with profanity filtered) or the original
            data if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Returning original data without filtering."
            )
            return data

        original_data = data
        filtered_data = data

        # Log a snippet of the data for debugging, handling potentially very long strings
        data_snippet = f"'{data[:100]}{'...' if len(data) > 100 else ''}'"
        logger.debug(f"[{self.node_name}] Starting profanity filtering for data: {data_snippet}")

        for pattern in self._compiled_patterns:
            filtered_data = pattern.sub(self._replacement_string, filtered_data)

        if original_data != filtered_data:
            # Log filtered data snippet for verification, if changes occurred
            original_snippet = f"'{original_data[:100]}{'...' if len(original_data) > 100 else ''}'"
            filtered_snippet = f"'{filtered_data[:100]}{'...' if len(filtered_data) > 100 else ''}'"
            logger.info(
                f"[{self.node_name}] Profanity filtered. "
                f"Original (first 100 chars): {original_snippet}. "
                f"Filtered (first 100 chars): {filtered_snippet}"
            )
        else:
            logger.debug(f"[{self.node_name}] No profanity found or filtered in data.")

        return filtered_data