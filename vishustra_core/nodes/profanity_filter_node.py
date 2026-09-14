import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node designed for filtering profanity from text data.

    This node processes input text, identifies predefined profane words, and
    replaces them with a specified replacement string (defaulting to '***').
    It performs case-insensitive matching and utilizes word boundaries to prevent
    unintended replacements of parts of non-profane words (e.g., 'hell' in 'hello').
    """

    def __init__(self, profane_words: List[str] = None, replacement_text: str = "***"):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profane_words (List[str], optional): A list of words to be considered profane.
                                                 If None, a default, internal list is used.
            replacement_text (str, optional): The string to replace profane words with.
                                              Defaults to '***'.
        """
        self._profane_words = [word.lower() for word in profane_words] if profane_words else self._default_profane_words()
        self._replacement_text = replacement_text
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._profane_words)} "
            f"profane words and replacement string '{self._replacement_text}'."
        )

    def _default_profane_words(self) -> List[str]:
        """
        Provides a default list of common profane words.

        This list is for demonstration and basic functionality. For a robust
        production system, it should be loaded from a comprehensive external
        configuration, database, or a specialized lexicon.
        """
        return ["badword", "cursedword", "damn", "hell", "ass", "bitch", "fuck", "shit", "cunt", "motherfucker", "bastard"]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out identified profane words.

        This method expects `data` to be a string. If `data` is not a string,
        it logs a warning and returns the data unchanged. Profane words found
        within the string are replaced with the configured `replacement_text`.

        Args:
            data (Any): The input data to be processed, ideally a string.
            context (Dict[str, Any]): The execution context dictionary.
                                      This node does not directly use context,
                                      but it's part of the `BaseNode` interface.

        Returns:
            Any: The filtered string if the input was a string and profanity was processed,
                 otherwise the original data (if not a string or an error occurred).
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data for processing. "
                f"Expected type 'str', got '{type(data).__name__}'. Returning data unchanged."
            )
            return data

        original_data = data
        filtered_text = data
        try:
            for word in self._profane_words:
                # Construct a regex pattern for whole word matching, ensuring case-insensitivity.
                # `re.escape(word)` handles any special regex characters within the profane word.
                # `r'\b'` asserts a word boundary, preventing partial word matches (e.g., 'hell' in 'hello').
                pattern = r'\b' + re.escape(word) + r'\b'
                # `re.sub` replaces all non-overlapping occurrences of the pattern.
                # `re.IGNORECASE` flag makes the matching case-insensitive.
                filtered_text = re.sub(pattern, self._replacement_text, filtered_text, flags=re.IGNORECASE)
                logger.debug(f"[{self.node_name}] Attempted to filter word '{word}'.")
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during profanity filtering: {e}",
                exc_info=True
            )
            # In case of an unexpected error, return the original data to prevent accidental data loss.
            return original_data

        if filtered_text != original_data:
            logger.info(f"[{self.node_name}] Profanity detected and filtered in text.")
        else:
            logger.debug(f"[{self.node_name}] No profanity detected in text.")

        return filtered_text