import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node that filters out common profanities from text data.
    It supports filtering individual strings or lists of strings.
    The filtering is case-insensitive and attempts to respect word boundaries.
    """

    # In a production system, this list would be dynamically loaded from a
    # configuration service, database, or a dedicated NLP library.
    # Using raw strings with word boundaries for basic regex matching.
    _PROFANITY_LIST_PATTERNS = [
        r"\bbadword1\b",
        r"\bswearword\b",
        r"\bcurse\b",
        r"\bprofane\b",
        r"\bfoul language\b", # Example phrase
    ]

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilter"

    def _filter_single_string(self, text: str) -> str:
        """
        Helper method to filter profanity from a single string.
        Uses regex for case-insensitive matching and word boundary awareness.
        """
        filtered_text = text
        for pattern in self._PROFANITY_LIST_PATTERNS:
            # Replace all occurrences of the pattern with '***' using case-insensitive matching.
            # re.sub is used for its robust regex capabilities.
            filtered_text = re.sub(pattern, "***", filtered_text, flags=re.IGNORECASE)
        return filtered_text

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[str, List[str]]:
        """
        Processes the input data to remove profanity.

        If `data` is a string, it filters the string.
        If `data` is a list of strings, it filters each string in the list.
        Non-string elements found within a list will be passed through
        unmodified, with a warning logged.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for the processing.
                     (Currently not used by this node but available for future extensions).

        Returns:
            The data with profanities filtered. The return type matches the
            input type (string or list of strings).

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
        """
        logger.debug(f"ProfanityFilterNode received data of type: {type(data).__name__}")

        if isinstance(data, str):
            result = self._filter_single_string(data)
            logger.info("Successfully filtered profanity from a single string.")
            return result
        elif isinstance(data, list):
            filtered_list = []
            for i, item in enumerate(data):
                if isinstance(item, str):
                    filtered_list.append(self._filter_single_string(item))
                else:
                    # Log a warning for non-string items in the list, but pass them through
                    # to avoid halting processing of other valid items.
                    logger.warning(
                        f"ProfanityFilterNode found non-string item at index {i} "
                        f"(type: {type(item).__name__}) in input list. "
                        "Item passed through unmodified."
                    )
                    filtered_list.append(item)
            logger.info(f"Successfully filtered profanity from {len(filtered_list)} items in a list.")
            return filtered_list
        else:
            # Raise an error for unsupported data types.
            error_msg = (
                f"{self.node_name} expects input 'data' to be a 'str' or 'List[str]', "
                f"but received type: '{type(data).__name__}'. Data: {data!r}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)