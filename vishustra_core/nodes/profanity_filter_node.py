import logging
import re
from typing import Any, Dict, List, Optional

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters profane words from input text.

    This node replaces specified profane words with a designated replacement
    character (e.g., asterisks). It supports case-insensitive detection by default.
    """

    def __init__(
        self,
        profane_words: Optional[List[str]] = None,
        replacement_char: str = "*",
        case_sensitive: bool = False
    ):
        """
        Initializes the ProfanityFilterNode with a list of words to filter.

        Args:
            profane_words (Optional[List[str]]): A list of words considered profane.
                                                 If None, a default curated list will be used.
            replacement_char (str): The character to use for replacing profane words.
                                    Must be a single character. Defaults to '*'.
            case_sensitive (bool): If True, profanity detection will be case-sensitive.
                                   Defaults to False (case-insensitive).
        Raises:
            ValueError: If `replacement_char` is not a single character string.
            TypeError: If `profane_words` is not a list or contains non-string elements.
        """
        if profane_words is None:
            # A common, small default list for demonstration. In a real system,
            # this might be loaded from a configuration file or external service.
            self._profane_words = ["fuck", "shit", "bitch", "asshole", "cunt", "damn"]
            logger.debug(f"[{self.node_name}] Using default profane words list.")
        elif not isinstance(profane_words, list) or not all(isinstance(w, str) for w in profane_words):
            logger.error(f"[{self.node_name}] `profane_words` must be a list of strings, but got {type(profane_words).__name__}.")
            raise TypeError("`profane_words` must be a list of strings.")
        else:
            self._profane_words = profane_words
            logger.debug(f"[{self.node_name}] Custom profane words list provided.")

        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            logger.error(f"[{self.node_name}] `replacement_char` must be a single character string, but got '{replacement_char}'.")
            raise ValueError("`replacement_char` must be a single character string.")
        self._replacement_char = replacement_char
        self._case_sensitive = case_sensitive

        # Pre-compile regex for efficiency. This ensures the pattern is built only once.
        if self._profane_words:
            # Create a regex pattern to match any of the profane words as whole words (\b).
            # re.escape is used to handle special characters in profane words safely.
            # The flags handle case-sensitivity.
            flags = 0 if case_sensitive else re.IGNORECASE
            self._profanity_pattern = re.compile(
                r'\b(?:' + '|'.join(re.escape(word) for word in self._profane_words) + r')\b',
                flags=flags
            )
            logger.debug(f"[{self.node_name}] Profanity pattern compiled: '{self._profanity_pattern.pattern}' (case_sensitive={case_sensitive})")
        else:
            self._profanity_pattern = None
            logger.warning(f"[{self.node_name}] Node initialized with an empty list of profane words. No filtering will occur.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profane words from the input data.

        This method expects the input `data` to be a string. If `data` is not a string,
        a `TypeError` is raised as profanity filtering is inherently a text operation.
        Profane words detected are replaced with the configured `replacement_char`.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. This node does not modify
                                       or directly use the context.

        Returns:
            Any: The processed data with profane words replaced. The return type
                 will match the input `data` type if it was a string.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        logger.debug(f"[{self.node_name}] Attempting to process data of type: {type(data).__name__}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Received non-string data (type: '{type(data).__name__}'). "
                "Profanity filtering is only applicable to strings."
            )
            raise TypeError(
                f"[{self.node_name}] Expected input `data` to be a string, "
                f"but received {type(data).__name__}."
            )

        if not self._profanity_pattern:
            logger.info(f"[{self.node_name}] No profane words configured or pattern is empty. Returning data without filtering.")
            return data

        original_data = data
        
        # Define the replacement function for re.sub.
        # It takes a match object and returns a string of `_replacement_char`
        # repeated `len(match.group(0))` times, ensuring the replacement
        # maintains the original length of the profane word.
        def replacer(match: re.Match) -> str:
            return self._replacement_char * len(match.group(0))

        filtered_data = self._profanity_pattern.sub(replacer, data)

        if filtered_data != original_data:
            logger.info(f"[{self.node_name}] Successfully filtered profanity from data.")
            # Log a snippet of the transformation for debugging without exposing full content
            log_original = original_data[:100] + ('...' if len(original_data) > 100 else '')
            log_filtered = filtered_data[:100] + ('...' if len(filtered_data) > 100 else '')
            logger.debug(f"[{self.node_name}] Original (snippet): '{log_original}'")
            logger.debug(f"[{self.node_name}] Filtered (snippet): '{log_filtered}'")
        else:
            logger.debug(f"[{self.node_name}] No profanity detected in data.")

        return filtered_data