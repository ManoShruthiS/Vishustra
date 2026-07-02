import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out specified profanity words from input text.

    This node supports filtering single strings, lists of strings, or string values
    within a dictionary. For non-string data types, the node logs a warning and
    passes the data through unchanged, ensuring robustness within an orchestration flow.
    """

    def __init__(self, profanity_list: Union[List[str], None] = None, replacement_char: str = '*'):
        """
        Initializes the ProfanityFilterNode with an optional custom profanity list
        and a character for replacement.

        Args:
            profanity_list: An optional list of strings to be recognized as profanity.
                            If None, a sensible default list is utilized.
            replacement_char: The character used to replace each letter of a detected
                              profane word. Defaults to '*'.
        """
        # A default, commonly recognized set of profanity words (for illustrative purposes)
        _default_profanity_set = {
            "shit", "fuck", "bitch", "asshole", "damn", "cunt",
            "pussy", "dick", "bastard", "motherfucker", "fucker", "prick"
        }

        # Use the provided list or the default, ensuring all words are lowercase and in a set for efficiency
        self._profanity_words = set(word.lower() for word in profanity_list) if profanity_list else _default_profanity_set
        self._replacement_char = replacement_char

        # Pre-compile regex patterns for each profanity word for efficient, case-insensitive,
        # and whole-word matching during processing. Each pattern is paired with its
        # corresponding replacement string.
        self._compiled_patterns = []
        for word in self._profanity_words:
            # Create a replacement string of the same length as the word, using the specified character
            replacement_string = self._replacement_char * len(word)
            # Compile regex for whole word (\b), case-insensitive matching (re.IGNORECASE)
            # re.escape is used to handle potential special regex characters within profanity words
            self._compiled_patterns.append(
                (re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE), replacement_string)
            )

        logger.info(f"ProfanityFilterNode initialized with {len(self._profanity_words)} words configured for filtering.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilterNode"

    def _filter_text(self, text: str) -> str:
        """
        Internal method to apply the profanity filter logic to a single string.
        """
        filtered_text = text
        for pattern, replacement_string in self._compiled_patterns:
            # Substitute all occurrences of the profane word with its replacement string
            filtered_text = pattern.sub(replacement_string, filtered_text)
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out any detected profanity.

        The method handles various input data types:
        - If `data` is a string, it is directly filtered.
        - If `data` is a list, each string item within the list is filtered. Other item types are passed through.
        - If `data` is a dictionary, string values are filtered. Other value types are passed through.
        - If `data` is `None`, it is returned as `None`.
        - For any other unsupported data type, a warning is logged, and the original data is returned
          unchanged to maintain flow continuity.

        Args:
            data: The input data to be processed. Expected types are `str`, `List[str]`, or `Dict[str, Any]`.
            context: A dictionary containing contextual information relevant to the current processing flow.
                     This node does not directly utilize the context but adheres to the signature.

        Returns:
            The processed data with profanity filtered. The type of the returned data
            matches the type of the input `data` where possible.
        """
        logger.debug(f"[{self.node_name}] Initiating processing for data of type: {type(data)}")

        if isinstance(data, str):
            processed_data = self._filter_text(data)
            if processed_data != data:
                logger.info(f"[{self.node_name}] Profanity detected and filtered in a string.")
            return processed_data
        elif isinstance(data, list):
            processed_list = []
            filtered_item_count = 0
            for item in data:
                if isinstance(item, str):
                    filtered_item = self._filter_text(item)
                    if filtered_item != item:
                        filtered_item_count += 1
                    processed_list.append(filtered_item)
                else:
                    processed_list.append(item) # Pass non-string items through untouched
            if filtered_item_count > 0:
                logger.info(f"[{self.node_name}] Filtered profanity in {filtered_item_count} list items out of {len(data)}.")
            return processed_list
        elif isinstance(data, dict):
            processed_dict = {}
            filtered_value_count = 0
            for key, value in data.items():
                if isinstance(value, str):
                    filtered_value = self._filter_text(value)
                    if filtered_value != value:
                        filtered_value_count += 1
                    processed_dict[key] = filtered_value
                else:
                    processed_dict[key] = value # Pass non-string values through untouched
            if filtered_value_count > 0:
                logger.info(f"[{self.node_name}] Filtered profanity in {filtered_value_count} dictionary values.")
            return processed_dict
        elif data is None:
            logger.debug(f"[{self.node_name}] Received None data. Returning None as is.")
            return None
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported data type for profanity filtering: '{type(data).__name__}'. "
                "Data will be returned unchanged to prevent flow disruption."
            )
            return data