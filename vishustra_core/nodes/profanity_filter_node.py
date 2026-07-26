import logging
import re
from typing import Any, Dict, List, Union

# Assuming this import path is correct within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanity from text data.
    It supports processing single strings or lists of strings, replacing identified
    profane words with a configurable replacement string.
    """

    # A class-level set of common profane words for quick lookup and efficient memory usage.
    # Words are kept lowercase for case-insensitive matching.
    _PROFANE_WORDS = {
        "fuck", "shit", "bitch", "asshole", "cunt", "damn", "hell",
        "piss", "bastard", "motherfucker", "cock", "dick", "prick",
        "wanker", "bollocks", "tits", "fag", "nigger", "slut", "whore"
    }

    def __init__(self, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode.

        Args:
            replacement_string (str, optional): The string to use for replacing
                                                profane words. Defaults to "***".
        """
        self._replacement_string = replacement_string

        # Pre-compile regex patterns for each profane word.
        # This improves performance, especially if the node processes a lot of data.
        # '\b' ensures whole word matching, and re.IGNORECASE handles case-insensitivity.
        self._profanity_patterns = {
            word: re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self._PROFANE_WORDS
        }
        logger.info(f"ProfanityFilterNode initialized with replacement string: '{self._replacement_string}'")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def _filter_single_string(self, text: str) -> str:
        """
        Filters profanity from a single string.

        Args:
            text (str): The input string to be filtered.

        Returns:
            str: The filtered string with profane words replaced.
        """
        filtered_text = text
        for word, pattern in self._profanity_patterns.items():
            # Apply the pre-compiled regex pattern to replace each profane word.
            filtered_text = pattern.sub(self._replacement_string, filtered_text)
        return filtered_text

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[str, List[str]]:
        """
        Processes the input data, filtering out profanity from text content.

        This method supports processing either a single string or a list of strings.
        If a list contains non-string items, they are passed through unchanged
        while string items are filtered. Unsupported data types are returned as-is
        with a warning logged.

        Args:
            data (Union[str, List[str]]): The input data to be processed.
                                          Expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing task (e.g., global settings,
                                      user preferences, etc.).

        Returns:
            Union[str, List[str]]: The data with profanity filtered, or the original
                                   data if it's not a string or list of strings.
        """
        logger.debug(f"ProfanityFilterNode received data for processing. Context keys: {list(context.keys())}")

        try:
            if isinstance(data, str):
                processed_data = self._filter_single_string(data)
                logger.debug(f"Processed single string (truncated): '{data[:50]}...' -> '{processed_data[:50]}...'")
                return processed_data
            elif isinstance(data, list):
                # Check if all items are strings for more specific logging,
                # but process only string items robustly.
                if not all(isinstance(item, str) for item in data):
                    logger.warning(
                        "ProfanityFilterNode received a list containing non-string items. "
                        "Only string items will be processed; others will be returned unchanged."
                    )
                processed_data = [
                    self._filter_single_string(item) if isinstance(item, str) else item
                    for item in data
                ]
                # Log a snippet of the first item if the list is not empty.
                if data:
                    logger.debug(f"Processed list of strings (first item truncated): '{data[0][:50]}...' -> '{processed_data[0][:50]}...'")
                else:
                    logger.debug("Processed an empty list of strings.")
                return processed_data
            else:
                logger.warning(
                    f"ProfanityFilterNode received unsupported data type: {type(data).__name__}. "
                    "Expected 'str' or 'List[str]'. Returning data unchanged."
                )
                return data
        except Exception as e:
            # Log the full traceback for unexpected errors during processing
            logger.exception(f"An unexpected error occurred during profanity filtering: {e}")
            # In case of an error, it's often best practice for a processing node
            # to return the original data or an indicator of failure, depending
            # on the orchestration framework's error handling strategy.
            # Here, we opt for returning the original data to ensure pipeline resilience.
            return data