
import logging
import re
from typing import Any, Dict, List, Set

# Ensure this import path is correct as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out offensive language from text data.
    It identifies predefined profane words and replaces them with a censored string
    (e.g., asterisks) while preserving word length and attempting case-insensitivity.
    """

    def __init__(self, replacement_char: str = '*', custom_profane_words: List[str] = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            replacement_char (str): The character used to replace each letter of an identified
                                    profane word. Defaults to '*'. Must be a single character.
            custom_profane_words (List[str], optional): A list of additional words to consider
                                                         profane. These will be added to a
                                                         default list. Words are processed
                                                         case-insensitively.
        """
        node_name_for_logging = self.node_name if hasattr(self, 'node_name') else "ProfanityFilterNode"

        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            logger.error(
                "[%s] Invalid 'replacement_char' '%s'. Must be a single character string. Defaulting to '*'.",
                node_name_for_logging,
                replacement_char
            )
            self._replacement_char = '*'
        else:
            self._replacement_char = replacement_char

        # Default list of profane words. Stored in lowercase for consistent internal processing.
        _default_profane_words: Set[str] = {
            "ass", "bastard", "bitch", "crap", "darn", "damn", "dick", "fag", "fuck",
            "goddamn", "hell", "piss", "prick", "shit", "slut", "son of a bitch", "whore",
            "arse", "bollocks", "bugger", "choad", "cock", "cunt", "feck", "gash", "jizz",
            "knob", "minge", "motherfucker", "nigga", "nigger", "paki", "punani", "pussy",
            "queer", "spastic", "tits", "twat", "wank", "wetback", "wtf", "lmao", "rofl"
        }
        
        all_profane_words: Set[str] = set(_default_profane_words)

        if custom_profane_words is not None:
            if not isinstance(custom_profane_words, list):
                logger.warning(
                    "[%s] 'custom_profane_words' must be a list of strings. Received type '%s'. Ignoring custom words.",
                    node_name_for_logging,
                    type(custom_profane_words).__name__
                )
            else:
                for word in custom_profane_words:
                    if isinstance(word, str):
                        all_profane_words.add(word.lower())
                    else:
                        logger.warning(
                            "[%s] Custom profane word item '%s' (type %s) is not a string. Ignoring it.",
                            node_name_for_logging,
                            word, type(word).__name__
                        )

        # Sort words by length in descending order. This can slightly optimize regex operations
        # by ensuring longer, more specific patterns are tried before shorter, generic ones,
        # though regex word boundaries significantly mitigate common substring issues.
        self._profane_words = sorted(list(all_profane_words), key=len, reverse=True)
        
        logger.debug(
            "[%s] Initialized with %d unique profane words and replacement char '%s'.",
            node_name_for_logging,
            len(self._profane_words), self._replacement_char
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out identified profane words.

        If the input `data` is a string, it replaces identified profane words
        with a string of `replacement_char` characters, matching the length
        of the original profane word and preserving word boundaries and case-insensitivity.

        If `data` is not a string, it logs a warning and returns the data unchanged,
        as profanity filtering is only applicable to text.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing. This node does not
                                       currently use the context for its core logic,
                                       but it is available for future extensions.

        Returns:
            Any: The processed data (a sanitized string if input was a string),
                 or the original data if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Received non-string data of type '%s'. Profanity filter applies only to strings. Returning data unchanged.",
                self.node_name, type(data).__name__
            )
            return data

        processed_text = data
        
        # Log a preview of the input data for debugging purposes
        log_data_preview = data[:100] + ('...' if len(data) > 100 else '')
        logger.debug("[%s] Starting profanity filtering for input text (preview: '%s').",
                     self.node_name, log_data_preview)

        words_filtered_count = 0
        for word in self._profane_words:
            # Create a replacement string of appropriate length
            replacement = self._replacement_char * len(word)
            
            # Use regex with word boundaries (\b) and case-insensitivity (re.IGNORECASE)
            # re.escape() is used to treat the word literally, in case it contains regex special characters.
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Apply the replacement and count how many times it occurred
            processed_text, count = re.subn(pattern, replacement, processed_text, flags=re.IGNORECASE)
            words_filtered_count += count

        logger.info(
            "[%s] Profanity filtering completed. Original length: %d, Processed length: %d. Total words filtered: %d.",
            self.node_name, len(data), len(processed_text), words_filtered_count
        )
        return processed_text

