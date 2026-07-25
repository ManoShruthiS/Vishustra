import re
import logging
from typing import Any, Dict, List, Union

# Assuming 'vishustra_core' is an installable package or properly configured in PYTHONPATH
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters profanity from text data.

    This node identifies predefined profanity words in string inputs and replaces
    them with a specified replacement string (default: '***'). It supports
    filtering individual strings, lists of strings, or string values within
    dictionaries (default key: 'text').
    """

    def __init__(self, profanity_list: Union[List[str], None] = None, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode with an optional custom profanity list
        and/or a custom replacement string.

        Args:
            profanity_list (List[str], optional): A list of profanity words to filter.
                                                  If None, a default list is used.
            replacement_string (str, optional): The string to replace profanity with.
                                                Defaults to '***'.
        """
        # Convert all profanity words to lowercase for case-insensitive matching
        self._profanity_list = [word.lower() for word in profanity_list] if profanity_list else [
            "shit", "fuck", "damn", "bitch", "asshole", "cunt", "motherfucker", "bastard"
        ]
        self._replacement_string = replacement_string

        # Compile regex patterns for efficient, case-insensitive, whole-word matching.
        # Sorting by length descending helps ensure longer, more specific profanities
        # (e.g., "asshole") are matched before shorter, contained ones ("ass").
        sorted_profanity = sorted(self._profanity_list, key=len, reverse=True)
        self._profanity_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in sorted_profanity
        ]
        logger.debug(f"ProfanityFilterNode initialized with profanity list: {self._profanity_list}")
        logger.debug(f"Replacement string set to: '{self._replacement_string}'")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def _filter_text(self, text: str) -> str:
        """Helper method to apply profanity filter to a single string."""
        if not isinstance(text, str):
            logger.warning(f"Expected input to _filter_text to be a string, but received {type(text)}. Returning as is.")
            return text

        original_text = text
        filtered_text = text
        for pattern in self._profanity_patterns:
            filtered_text = pattern.sub(self._replacement_string, filtered_text)
        
        if original_text != filtered_text:
            logger.debug(f"Text filtered. Original (partial): '{original_text[:50]}...', Filtered (partial): '{filtered_text[:50]}...'")
        
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out predefined profanity words.

        The method handles different data types:
        - If `data` is a string, it filters the string directly.
        - If `data` is a list, it attempts to filter each item as a string.
        - If `data` is a dictionary, it looks for a string value at the key
          specified by `context.get('target_key', 'text')` and filters that value.
          If the value is a list, it filters each item in the list.
        - For other data types, it logs a warning and returns the data unchanged.

        Args:
            data (Any): The input data to be processed. Expected types are `str`,
                        `list[str]`, or `dict` containing `str` or `list[str]` values.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       Can include 'target_key' (str) to specify which
                                       key in a dictionary should be filtered.

        Returns:
            Any: The processed data with profanity filtered. The structure of the
                 returned data matches the input data type.
        
        Raises:
            Exception: Re-raises any unexpected exceptions encountered during processing.
        """
        logger.info(f"ProfanityFilterNode received data of type: {type(data)}")

        try:
            if isinstance(data, str):
                return self._filter_text(data)

            elif isinstance(data, list):
                return [self._filter_text(item) for item in data]

            elif isinstance(data, dict):
                target_key = context.get('target_key', 'text')
                if target_key in data:
                    value_to_filter = data[target_key]
                    processed_data = data.copy() # Create a shallow copy to avoid modifying original dict

                    if isinstance(value_to_filter, str):
                        processed_data[target_key] = self._filter_text(value_to_filter)
                        return processed_data
                    elif isinstance(value_to_filter, list):
                        processed_data[target_key] = [self._filter_text(item) for item in value_to_filter]
                        return processed_data
                    else:
                        logger.warning(
                            f"Dict value at key '{target_key}' is of unsupported type '{type(value_to_filter)}'. "
                            "Expected str or list[str]. Returning original dict."
                        )
                        return data
                else:
                    logger.warning(
                        f"Dict data received, but specified target_key '{target_key}' not found. "
                        "Returning original dict without filtering."
                    )
                    return data
            else:
                logger.warning(
                    f"Unsupported data type for ProfanityFilterNode: {type(data)}. "
                    "Expected str, list, or dict. Returning data as is."
                )
                return data

        except Exception as e:
            logger.exception(f"An unexpected error occurred during profanity filtering: {e}")
            # Re-raise the exception to propagate the error up the orchestration chain.
            raise