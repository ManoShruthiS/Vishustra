import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A processing node that filters out profane words from input text.
    It replaces detected profanities with a sequence of asterisks,
    preserving the length of the original word.
    """

    _DEFAULT_PROFANITIES: Set[str] = {
        "shit", "fuck", "bitch", "asshole", "damn", "cunt", "motherfucker",
        "bastard", "dick", "pussy", "hell", "bollocks", "wanker"
    }
    _REPLACEMENT_CHAR: str = "*"

    def __init__(self, profane_words: List[str] = None):
        """
        Initializes the ProfanityFilterNode with an optional custom list of profane words.
        If no words are provided, a default set of common profanities is used.

        Args:
            profane_words: An optional list of strings considered profane.
        """
        self._profane_words: Set[str] = set(word.lower() for word in profane_words) if profane_words else self._DEFAULT_PROFANITIES
        if not self._profane_words:
            logger.warning("[ProfanityFilterNode] Initialized with an empty list of profane words. No filtering will occur.")
        else:
            logger.debug(f"[{self.node_name}] Initialized with {len(self._profane_words)} profane words.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out profane words.

        Expects the input `data` to be a string. If `data` is not a string,
        a warning is logged, and the original data is returned unmodified.
        Profane words are replaced with asterisks of the same length.

        Args:
            data: The input data, expected to be a string for filtering.
            context: A dictionary for shared context or state across nodes.
                     Not directly used by this node's core logic but provided
                     as part of the standard node interface.

        Returns:
            The processed data with profane words filtered, or the original data
            if it was not a string or no profanities were found.

        Raises:
            TypeError: While the current implementation handles non-string data
                       by logging a warning and returning, a more strict
                       implementation could raise TypeError here.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data).__name__}. "
                "Profanity filtering will be skipped. Returning original data."
            )
            return data

        if not self._profane_words:
            logger.debug(f"[{self.node_name}] No profane words configured. Returning original data.")
            return data

        processed_text = data
        found_matches: List[str] = []

        # Construct a regex pattern for all profane words, case-insensitive, with word boundaries.
        # This ensures 'ass' in 'classify' is not matched.
        # Each word is escaped to handle special regex characters if present.
        pattern_str = r'\b(' + '|'.join(re.escape(word) for word in self._profane_words) + r')\b'
        profanity_pattern = re.compile(pattern_str, re.IGNORECASE)

        def replacer(match: re.Match) -> str:
            """
            Callback function for re.sub to replace found words.
            Records the found word and returns an asterisk string of matching length.
            """
            matched_word = match.group(0)
            found_matches.append(matched_word)
            return self._REPLACEMENT_CHAR * len(matched_word)

        processed_text = profanity_pattern.sub(replacer, data)

        if found_matches:
            # Log detected profanities, converting them to lowercase for consistency
            unique_found_words = sorted(list(set(w.lower() for w in found_matches)))
            log_msg = f"[{self.node_name}] Filtered profanity in text. Detected words: {', '.join(unique_found_words)}"
            if len(log_msg) > 500: # Truncate long log messages for readability
                log_msg = log_msg[:497] + "..."
            logger.info(log_msg)
        else:
            logger.debug(f"[{self.node_name}] No profanity detected in text.")

        return processed_text
