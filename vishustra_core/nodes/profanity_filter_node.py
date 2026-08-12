import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

# Initialize a logger for this module
logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out profane words from input text.
    It replaces specified profane words with a designated censor character,
    maintaining the original word's length for readability and context.

    This node leverages regular expressions for robust, case-insensitive,
    and whole-word matching to prevent partial word censorship.
    """

    def __init__(self, profane_words: Optional[List[str]] = None, censor_char: str = '*'):
        """
        Initializes the ProfanityFilterNode with a list of words to filter
        and a character to use for censoring.

        Args:
            profane_words: An optional list of strings considered profane.
                           If None or an empty list, a default set of words will be used.
                           Words will be converted to lowercase internally for case-insensitive matching.
            censor_char: The single character to use for replacing profane words.
                         Defaults to '*'.
        """
        if profane_words is None or not profane_words:
            self._profane_words = self._get_default_profane_words()
            logger.info(f"[{self.node_name}] Initialized with default profane word list.")
        else:
            self._profane_words = [word.lower() for word in profane_words]
            logger.info(f"[{self.node_name}] Initialized with {len(self._profane_words)} custom profane words.")

        if not isinstance(censor_char, str) or len(censor_char) != 1:
            logger.warning(
                f"[{self.node_name}] Invalid censor_char '{censor_char}'. "
                "Using default '*' character."
            )
            self._censor_char = '*'
        else:
            self._censor_char = censor_char

        logger.debug(
            f"[{self.node_name}] Configuration: {len(self._profane_words)} words, "
            f"censor_char='{self._censor_char}'"
        )

    def _get_default_profane_words(self) -> List[str]:
        """
        Provides a default list of common profane words.
        In a production environment, this list might be loaded from a configuration
        file, a database, or an external service.
        """
        return ["fuck", "shit", "asshole", "bitch", "cunt", "damn", "piss", "bastard"]

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by identifying and censoring profane words.

        The method expects `data` to be a string. If `data` is not a string,
        it logs a warning and returns the original data unchanged.
        The `context` dictionary is currently not used by this node but is
        available for future extensions, e.g., for dynamic word lists.

        Args:
            data: The input data, ideally a string that needs filtering.
            context: A dictionary containing contextual information relevant
                     to the current processing pipeline.

        Returns:
            The processed data with profane words replaced by the censor character,
            or the original data if it was not a string.
        """
        logger.info(f"[{self.node_name}] Initiating profanity filtering process.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data type ({type(data).__name__}) is not a string. "
                "Profanity filtering will be skipped."
            )
            return data

        processed_text = data

        for word_to_censor in self._profane_words:
            # Construct a regex pattern for whole-word, case-insensitive matching
            # re.escape() handles special characters in the word
            # \b ensures word boundaries
            pattern = r'\b' + re.escape(word_to_censor) + r'\b'

            # Create the replacement string (e.g., '****' for a 4-letter word)
            censor_replacement = self._censor_char * len(word_to_censor)

            # Perform the replacement using re.sub for case-insensitivity
            # and global replacement
            if re.search(pattern, processed_text, re.IGNORECASE):
                processed_text = re.sub(pattern, censor_replacement, processed_text, flags=re.IGNORECASE)
                logger.debug(f"[{self.node_name}] Censored instances of '{word_to_censor}'.")

        logger.info(f"[{self.node_name}] Profanity filtering process completed.")
        return processed_text