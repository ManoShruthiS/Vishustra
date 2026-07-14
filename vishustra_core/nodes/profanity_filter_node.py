import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out profane words from input text.

    This node replaces detected profane words with a specified replacement character,
    maintaining the original length of the censored word. It handles case-insensitivity
    and uses word boundaries to prevent partial word censoring (e.g., "assessment" vs "ass").
    """

    def __init__(self,
                 profane_words: Optional[List[str]] = None,
                 replacement_char: str = '*') -> None:
        """
        Initializes the ProfanityFilterNode with a list of profane words and a
        replacement character.

        Args:
            profane_words (Optional[List[str]]): A list of words to be filtered.
                                                 If None, a default list of common
                                                 profanities is used.
            replacement_char (str): The single character to use for replacing
                                    profane words. Defaults to '*'.
        
        Raises:
            TypeError: If `profane_words` is not a list of strings.
            ValueError: If `replacement_char` is not a single character string.
        """
        # Default list for common profanities if none provided
        self._profane_words = [
            'fuck', 'shit', 'asshole', 'bitch', 'cunt', 'damn', 'piss', 'bastard',
            'motherfucker', 'cock', 'dick', 'wanker', 'bollocks', 'prick'
        ] if profane_words is None else profane_words
        
        if not isinstance(self._profane_words, list) or not all(isinstance(w, str) for w in self._profane_words):
            logger.error(f"Initialization error: profane_words must be a list of strings, got {type(self._profane_words).__name__}.")
            raise TypeError("`profane_words` must be a list of strings.")

        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            logger.error(f"Initialization error: replacement_char must be a single character string, got '{replacement_char}'.")
            raise ValueError("`replacement_char` must be a single character string.")
        self._replacement_char = replacement_char
        
        # Pre-compile regex patterns for efficiency.
        # Uses word boundaries (\b) and case-insensitive matching.
        # re.escape() is used to handle special characters within profane words.
        self._patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self._profane_words
        ]
        
        logger.debug(
            f"ProfanityFilterNode initialized with {len(self._profane_words)} "
            f"profane words and replacement char '{self._replacement_char}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profane words.

        Expects `data` to be a string. If `data` is not a string, it logs an error
        and raises a TypeError, adhering to a strict contract.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for processing. (Not used by this node, but
                                       required by the `BaseNode` signature).

        Returns:
            str: The processed string with profane words filtered.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"ProfanityFilterNode received non-string data. "
                f"Expected `str`, but received {type(data).__name__}. "
                "Raising TypeError."
            )
            raise TypeError(
                f"ProfanityFilterNode: Input `data` must be a string, "
                f"but received {type(data).__name__}."
            )

        processed_text = data
        original_text_checksum = hash(data) # Simple check to see if text changed

        for pattern in self._patterns:
            # Use re.sub with a lambda function to replace matched words.
            # The lambda function ensures that the replacement string consists of
            # the `_replacement_char` repeated for the exact length of the matched word,
            # preserving string length.
            processed_text = pattern.sub(
                lambda m: self._replacement_char * len(m.group(0)),
                processed_text
            )
        
        if hash(processed_text) != original_text_checksum:
            logger.info(
                f"Profanity filter applied to data. "
                f"Original (excerpt): '{data[:50]}...', "
                f"Processed (excerpt): '{processed_text[:50]}...'"
            )
        else:
            logger.debug("No profanity detected in data.")

        return processed_text
