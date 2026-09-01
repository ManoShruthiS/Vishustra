import re
import logging
from typing import Any, Dict, List, Set

# Assuming BaseNode is located in the specified path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanities from text data.

    This node uses a configurable list of banned words and regular expressions
    to perform case-insensitive replacement of detected profanity with censor characters.
    Non-string inputs are safely passed through without modification.
    """

    DEFAULT_BANNED_WORDS: Set[str] = {
        "fuck", "shit", "asshole", "bitch", "cunt", "motherfucker", "bastard", "damn", "hell",
        "cock", "pussy", "dick", "fucker", "slut", "whore", "tits", "crap"
    }
    CENSOR_CHAR: str = "*"

    def __init__(self, banned_words: List[str] = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            banned_words (List[str], optional): A list of words to filter. If None or empty,
                                                a predefined default set of profanities is used.
                                                Words are converted to lowercase for case-insensitive matching.
        """
        if banned_words:
            if not isinstance(banned_words, list):
                raise TypeError("banned_words must be a list of strings.")
            self._banned_words: Set[str] = {word.lower() for word in banned_words if isinstance(word, str)}
            if not self._banned_words:
                logger.warning(
                    f"[{self.node_name}] Provided banned_words list was empty or contained only non-strings. "
                    "Falling back to default profanity list."
                )
                self._banned_words = self.DEFAULT_BANNED_WORDS
        else:
            self._banned_words = self.DEFAULT_BANNED_WORDS

        # Create a robust regex pattern for efficient and accurate replacement.
        # Words are sorted by length descending to ensure that longer words (e.g., "motherfucker")
        # are matched before their potential substrings (e.g., "fucker") if they were to overlap,
        # although word boundaries '\b' largely mitigate this specific issue.
        # re.escape() handles special characters in profanity terms if any.
        pattern_words = sorted(list(self._banned_words), key=len, reverse=True)
        if pattern_words:
            self._profanity_pattern = re.compile(
                r'\b(' + '|'.join(re.escape(word) for word in pattern_words) + r')\b',
                re.IGNORECASE
            )
            logger.debug(f"[{self.node_name}] Initialized with {len(self._banned_words)} banned words.")
        else:
            # This case should ideally not be reached if default words are used,
            # but a safeguard for custom empty lists.
            self._profanity_pattern = None
            logger.warning(f"[{self.node_name}] No profanity words configured. Node will pass data through unchanged.")


    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out profanities if the data is a string.

        If `data` is not a string, it is returned unchanged, and a warning is logged.
        The `context` dictionary is currently not utilized by this node but is
        included as part of the `BaseNode` interface.

        Args:
            data (Any): The input data to be processed. Expected to be a string for filtering.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the processing operation.

        Returns:
            Any: The processed data. If `data` was a string and contained profanity,
                 a censored string is returned. Otherwise, the original `data` is returned.
        """
        if self._profanity_pattern is None:
            logger.debug(f"[{self.node_name}] No profanity words configured, skipping filter.")
            return data

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering will be skipped for this item."
            )
            return data

        original_data = data
        filtered_data = self._profanity_pattern.sub(self._censor_match, data)

        if original_data != filtered_data:
            logger.info(f"[{self.node_name}] Profanity detected and filtered in text.")
        else:
            logger.debug(f"[{self.node_name}] No profanity found in text.")

        return filtered_data

    def _censor_match(self, match: re.Match) -> str:
        """
        Helper function used by `re.sub` to replace a matched profanity with censor characters.

        The replacement string consists of the `CENSOR_CHAR` repeated for the length
        of the matched profanity, preserving the original length of the text.
        """
        matched_word = match.group(0)
        return self.CENSOR_CHAR * len(matched_word)
