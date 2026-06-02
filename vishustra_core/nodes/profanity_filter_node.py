import logging
import re
from typing import Any, Dict, List, Optional, Set

# Assuming BaseNode is available at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out profanity from input text.

    It replaces specified profane words with a configurable replacement character,
    ensuring a cleaner output. The filtering is case-insensitive and respects
    word boundaries to avoid partial word replacement.
    """

    _DEFAULT_PROFANITY_LIST: Set[str] = {
        "ass", "bitch", "fuck", "shit", "damn", "crap", "idiot"
    }
    _DEFAULT_REPLACEMENT_CHAR: str = "*"

    def __init__(self,
                 profanity_list: Optional[List[str]] = None,
                 replacement_char: str = _DEFAULT_REPLACEMENT_CHAR):
        """
        Initializes the ProfanityFilterNode with a custom or default profanity list.

        Args:
            profanity_list (Optional[List[str]]): An optional list of words to filter.
                                                  If None, a default list is used.
                                                  Words are converted to lowercase internally
                                                  for case-insensitive matching.
            replacement_char (str): The character used to replace profane words.
                                    Must be a single character. Defaults to '*'.

        Raises:
            ValueError: If replacement_char is not a single character string.
        """
        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            raise ValueError("replacement_char must be a single character string.")

        self._profanity_words: Set[str] = {
            word.lower() for word in (profanity_list or self._DEFAULT_PROFANITY_LIST)
        }
        self._replacement_char: str = replacement_char
        logger.debug(f"ProfanityFilterNode initialized with {len(self._profanity_words)} profanity words and replacement char '{self._replacement_char}'.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profanity from text.

        If the input `data` is a string, this method replaces any occurrences
        of words from the configured profanity list with the `replacement_char`,
        matching the original word's length. The filtering is case-insensitive
        and uses word boundaries (`\\b`) to ensure only whole words are replaced.

        Non-string inputs will result in a `TypeError`.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.

        Returns:
            Any: The filtered string if `data` was a string.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"ProfanityFilterNode received non-string data of type {type(data).__name__}. "
                "Expected a string for filtering text."
            )
            raise TypeError("ProfanityFilterNode expects string input for data.")

        filtered_text = data
        for word in self._profanity_words:
            # Create a regex pattern to match the whole word (case-insensitive).
            # re.escape() is used to escape any special characters in the profanity word
            # to prevent them from being interpreted as regex metacharacters.
            pattern = r'\b' + re.escape(word) + r'\b'

            # Use re.sub with re.IGNORECASE flag for case-insensitive matching.
            # The 'repl' argument is a function that returns a string of the
            # replacement character repeated for the length of the matched word.
            filtered_text = re.sub(
                pattern,
                lambda match: self._replacement_char * len(match.group(0)),
                filtered_text,
                flags=re.IGNORECASE
            )

        # Log a snippet of the processed text for debugging/monitoring
        log_input_snippet = data[:75] + ("..." if len(data) > 75 else "")
        log_output_snippet = filtered_text[:75] + ("..." if len(filtered_text) > 75 else "")
        logger.info(f"ProfanityFilterNode processed text. Input: '{log_input_snippet}' -> Output: '{log_output_snippet}'")

        return filtered_text