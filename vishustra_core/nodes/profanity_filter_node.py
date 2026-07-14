import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanity from text data.
    It replaces detected profane words with asterisks or a specified replacement character.
    """

    # Default list of profane words. Can be extended or overridden during initialization.
    _DEFAULT_PROFANE_WORDS = [
        "fuck", "shit", "bitch", "asshole", "damn", "cunt", "bastard", "cock", "pussy", "dick",
        "motherfucker", "fucker", "prick", "whore"
    ]
    _DEFAULT_REPLACEMENT_CHAR = "*"

    def __init__(self, profane_words: List[str] = None, replacement_char: str = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profane_words (List[str], optional): A custom list of profane words to filter.
                                                  If None, uses a default list. Each word
                                                  will be converted to lowercase internally.
            replacement_char (str, optional): The character to use for replacing profane words.
                                               If None, uses '*'.
        """
        # Normalize profane words to lowercase and remove duplicates, then sort for consistency
        unique_profane_words = sorted(list(set(word.lower() for word in (profane_words or self._DEFAULT_PROFANE_WORDS))))
        self._profane_words = unique_profane_words
        self._replacement_char = replacement_char or self._DEFAULT_REPLACEMENT_CHAR

        # Compile regex patterns for efficient and robust word boundary matching.
        # re.escape() is used to treat special characters in words literally.
        # re.IGNORECASE makes the matching case-insensitive.
        self._profanity_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self._profane_words
        ]
        logger.debug(f"[{self.node_name}] Initialized with {len(self._profanity_patterns)} profanity patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profane words by replacing them
        with the configured replacement character.

        Args:
            data (Any): The input data, expected to be a string for filtering.
                        If not a string, the data is returned unchanged.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing pipeline. Not directly used
                                       by this node but passed along.

        Returns:
            Any: The processed data (a string with profanity filtered) or the original
                 data if it's not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data)}. "
                "Profanity filtering skipped. Returning original data."
            )
            return data

        processed_text = data
        # Truncate original text for logging to prevent excessively long log messages
        original_text_sample = data[:100] + ('...' if len(data) > 100 else '')

        filtered_count = 0
        for pattern in self._profanity_patterns:
            # Use a lambda function for replacement to dynamically create the replacement
            # string based on the length of the matched profane word and to count occurrences.
            def replacer(match):
                nonlocal filtered_count
                filtered_count += 1
                return self._replacement_char * len(match.group(0))

            # Apply replacement using the compiled regex pattern
            processed_text = pattern.sub(replacer, processed_text)

        if filtered_count > 0:
            # Truncate processed text for logging
            processed_text_sample = processed_text[:100] + ('...' if len(processed_text) > 100 else '')
            logger.info(
                f"[{self.node_name}] Applied profanity filter. "
                f"Detected and replaced {filtered_count} instances of profanity. "
                f"Original (sample): '{original_text_sample}', "
                f"Processed (sample): '{processed_text_sample}'"
            )
        else:
            logger.debug(f"[{self.node_name}] No profanity detected in data.")

        return processed_text