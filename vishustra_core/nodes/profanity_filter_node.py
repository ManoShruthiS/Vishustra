import logging
import re
from typing import Any, Dict, List, Union

# Simulate the import path from the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out profanity from text data.

    This node can process strings, dictionaries (by specifying a key for text),
    and lists containing strings or dictionaries.
    """

    # A simple, static list of profanities for demonstration purposes.
    # In a production environment, this list would typically be loaded from
    # a configurable source (e.g., database, external file, or an advanced NLP library)
    # and potentially be much more extensive and dynamic.
    _BAD_WORDS = [
        "fuck", "shit", "asshole", "bitch", "damn", "cunt", "prick",
        "wank", "bollocks", "motherfucker", "bastard"
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def _censor_word(self, word: str) -> str:
        """Replaces a word with asterisks of the same length, preserving case if possible."""
        # For simplicity, we just return asterisks of the same length.
        # A more advanced version might preserve first/last character or match case.
        return "*" * len(word)

    def _filter_text(self, text: str) -> str:
        """Applies profanity filtering to a given string."""
        if not isinstance(text, str):
            logger.warning(
                f"ProfanityFilterNode received non-string data for text filtering: {type(text)}. "
                "Returning original data."
            )
            return text

        processed_text = text
        for bad_word in self._BAD_WORDS:
            # Use regex to find whole words, ignoring case.
            # re.escape is crucial to handle special characters in bad_word correctly.
            pattern = re.compile(r'\b' + re.escape(bad_word) + r'\b', re.IGNORECASE)
            processed_text = pattern.sub(self._censor_word(bad_word), processed_text)
            
        return processed_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        The `data` can be:
        - A string: The string will be filtered directly.
        - A dictionary: The node will look for a string value at a specified
          key (configurable via `context['profanity_filter_key']`, defaulting
          to 'text' or 'message') and filter that value.
        - A list: The node will iterate through the list. If an item is a
          string or a dictionary, it will apply the respective filtering logic.
          Other types in the list will be passed through unchanged.
        - Other types: Will be returned as is, with a warning logged.

        Args:
            data (Any): The input data to be processed.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Optional key: 'profanity_filter_key' (str)
                                      to specify which key in a dictionary to filter.

        Returns:
            Any: The processed data with profanity filtered.
        """
        try:
            if isinstance(data, str):
                logger.debug("ProfanityFilterNode: Processing string data.")
                return self._filter_text(data)

            elif isinstance(data, dict):
                # Determine which key to filter based on context or common defaults
                filter_key = context.get('profanity_filter_key')
                if not filter_key:
                    if 'text' in data:
                        filter_key = 'text'
                    elif 'message' in data:
                        filter_key = 'message'
                    else:
                        logger.warning(
                            f"ProfanityFilterNode: Dictionary does not contain a common text key ('text', 'message') "
                            "and 'profanity_filter_key' was not provided in context. "
                            "Data will be returned unchanged."
                        )
                        return data

                if filter_key in data:
                    if isinstance(data[filter_key], str):
                        logger.debug(f"ProfanityFilterNode: Processing dictionary data with key '{filter_key}'.")
                        processed_data = data.copy() # Operate on a copy to avoid modifying original input
                        processed_data[filter_key] = self._filter_text(data[filter_key])
                        return processed_data
                    else:
                        logger.warning(
                            f"ProfanityFilterNode: Dictionary key '{filter_key}' found but its value "
                            f"is not a string (type: {type(data[filter_key])}). "
                            "Data will be returned unchanged."
                        )
                        return data
                else:
                    logger.warning(
                        f"ProfanityFilterNode: Dictionary does not contain the specified filter key '{filter_key}'. "
                        "Data will be returned unchanged."
                    )
                    return data

            elif isinstance(data, list):
                logger.debug("ProfanityFilterNode: Processing list data.")
                processed_list = []
                for idx, item in enumerate(data):
                    # Recursively call process for list items that are strings or dictionaries.
                    # This allows filtering of complex nested structures.
                    if isinstance(item, (str, dict)):
                        processed_list.append(self.process(item, context))
                    else:
                        # Keep non-string/non-dict items as is
                        processed_list.append(item)
                return processed_list

            else:
                logger.warning(
                    f"ProfanityFilterNode received unsupported data type: {type(data)}. "
                    "Data will be returned unchanged."
                )
                return data

        except Exception as e:
            logger.error(f"ProfanityFilterNode encountered an unhandled error during processing: {e}", exc_info=True)
            # Depending on the desired error handling strategy, we might re-raise the exception,
            # return the original data, or return a specific error object.
            # For this scenario, we return the original data and log the error.
            return data