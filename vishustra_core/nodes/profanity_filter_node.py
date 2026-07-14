import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Default list of profanity words (for demonstration purposes).
# In a production system, this list would typically be loaded from a configuration
# file, a database, or managed by a dedicated profanity detection service to be dynamic and extensive.
_DEFAULT_PROFANITY_LIST = [
    "damn", "shit", "fuck", "bitch", "asshole", "cunt", "motherfucker",
    "fucker", "bastard", "dick", "pussy", "tits", "cock", "prick", "wanker",
    "bollocks", "crap", "bugger", "hell"
]

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out profanity from text data.

    This node identifies specified profanity words within an input string and
    replaces them with a chosen replacement string or character. It supports
    case-insensitive matching and handles whole-word replacements, aiming to
    maintain readability while censoring offensive content.
    """

    def __init__(self, profanity_list: List[str] = None, replacement: str = '*', case_sensitive: bool = False):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list: An optional list of custom profanity words. If None,
                            a robust default list is used. Words are internally sorted
                            by length (descending) to ensure correct replacement
                            of overlapping terms (e.g., 'fucker' before 'fuck').
                            Non-string elements are converted to strings.
            replacement: The character or string used to mask profanity.
                         If a single character (e.g., '*'), it will be repeated
                         to match the length of the detected profanity word,
                         preserving string length. If a string (e.g., '[CENSORED]'),
                         it will be inserted as-is. Defaults to '*'.
            case_sensitive: A boolean flag. If True, filtering will strictly match
                            the casing of words in `profanity_list`. Defaults to False,
                            performing case-insensitive matching.
        """
        # Ensure profanity_list elements are strings and sort by length descending.
        # Sorting by length helps prevent partial matches (e.g., matching "fuck" inside "fucker").
        self._profanity_list = sorted([str(word) for word in (profanity_list or _DEFAULT_PROFANITY_LIST) if word],
                                      key=len, reverse=True)
        # Ensure replacement is a non-empty string.
        self._replacement = str(replacement) if replacement else '*'
        self._case_sensitive = bool(case_sensitive)

        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._profanity_list)} profanity words. "
            f"Replacement strategy: '{self._replacement}', Case-sensitive: {self._case_sensitive}"
        )
        if not self._profanity_list:
            logger.warning(f"[{self.node_name}] Node initialized with an empty profanity list. No words will be filtered.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, applying profanity filtering.

        The node expects `data` to be a string. It iterates through its configured
        profanity list, identifies occurrences of these words (respecting whole-word
        boundaries), and replaces them with the specified `replacement` string or
        repeated character.

        Args:
            data: The input data, which must be a string to be processed.
            context: A dictionary containing contextual information relevant to
                     the current processing pipeline run. Not directly used by
                     this node for filtering logic, but included as per `BaseNode`
                     interface for future extensibility.

        Returns:
            The processed string with profanity filtered out.

        Raises:
            TypeError: If the input `data` is not a string, as this node is
                       designed exclusively for string manipulation.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type for processing. "
                f"Expected 'str', but received '{type(data).__name__}'. "
                "This node requires string input to perform profanity filtering."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, "
                f"but received type '{type(data).__name__}'."
            )

        # Handle empty or whitespace-only strings early
        if not data.strip():
            logger.debug(f"[{self.node_name}] Received empty or whitespace-only string, returning as is.")
            return data

        # If the profanity list is empty, no filtering will occur
        if not self._profanity_list:
            logger.warning(f"[{self.node_name}] Profanity list is empty. Skipping filtering for data: '{data[:100]}'")
            return data
            
        processed_data = data
        filtered_occurrences = 0
        
        logger.debug(f"[{self.node_name}] Starting profanity filtering for input (first 100 chars): '{data[:100]}'")

        # Set regex flags based on the `case_sensitive` configuration
        flags = 0 if self._case_sensitive else re.IGNORECASE

        for profanity_word in self._profanity_list:
            # Construct a regex pattern for whole words using word boundaries (`\b`).
            # `re.escape` is crucial to handle profanity words that might contain
            # regex special characters (e.g., '.', '+', '*').
            pattern = re.compile(r'\b' + re.escape(profanity_word) + r'\b', flags)
            
            # Find all matches to accurately count occurrences before performing replacement.
            matches = list(pattern.finditer(processed_data))
            
            if matches:
                filtered_occurrences += len(matches)
                
                # Determine the actual replacement string based on configuration.
                if len(self._replacement) == 1:
                    # Repeat the single character to match the profanity word's length,
                    # maintaining the original string length.
                    actual_replacement_str = self._replacement * len(profanity_word)
                else:
                    # Use the provided replacement string as-is.
                    actual_replacement_str = self._replacement
                
                # Perform the replacement using the compiled pattern.
                processed_data = pattern.sub(actual_replacement_str, processed_data)
                
                logger.debug(
                    f"[{self.node_name}] Replaced '{profanity_word}' {len(matches)} time(s). "
                    f"Current data snippet (first 50 chars): '{processed_data[:50]}...'"
                )

        if filtered_occurrences > 0:
            logger.info(
                f"[{self.node_name}] Successfully filtered {filtered_occurrences} profanity "
                f"occurrence(s) in the input string. Final string (first 100 chars): '{processed_data[:100]}'"
            )
        else:
            logger.debug(f"[{self.node_name}] No profanity found in the input string.")

        return processed_data