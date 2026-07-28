import logging
import re
from typing import Any, Dict, List, Optional

# Assuming this path is correctly configured in the Vishustra project
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profane words
    from input text, replacing them with a configured character.

    This node is designed to sanitize text data by identifying and masking
    profane language based on a customizable list. It supports case-insensitive
    matching and whole-word replacement for robustness.
    """

    def __init__(self,
                 profanity_list: Optional[List[str]] = None,
                 replacement_char: str = '*',
                 case_sensitive: bool = False):
        """
        Initializes the ProfanityFilterNode with a list of profanities and replacement settings.

        Args:
            profanity_list: An optional list of words to be considered profane.
                            If None, a default, internal list is utilized.
            replacement_char: The character to use for masking profane words.
                              Each character of a detected profane word will be
                              replaced by this character (e.g., "badword" -> "*******").
                              Defaults to '*'.
            case_sensitive: If True, profanity matching will be case-sensitive.
                            Defaults to False, performing case-insensitive matching.
        """
        # A default, illustrative profanity list. In a production system, this
        # would typically be loaded from configuration, an external service,
        # or a comprehensive dataset.
        self._profanity_list = profanity_list if profanity_list is not None else [
            "fuck", "shit", "asshole", "bitch", "cunt", "damn", "motherfucker"
        ]
        self._replacement_char = replacement_char
        self._case_sensitive = case_sensitive

        # Pre-compile regex patterns for efficiency and robust matching.
        # This handles special characters in profane words and ensures whole word matching.
        self._compiled_patterns = []
        for word in self._profanity_list:
            flags = 0 if self._case_sensitive else re.IGNORECASE
            # \b ensures that only whole words are matched (e.g., "hell" matches 'hell' but not 'hello')
            self._compiled_patterns.append(re.compile(r'\b' + re.escape(word) + r'\b', flags))
        
        logger.debug(
            f"ProfanityFilterNode initialized with {len(self._profanity_list)} profanities "
            f"(case_sensitive={self._case_sensitive}) and replacement char '{replacement_char}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out and mask any detected profane words.

        Args:
            data: The input data, expected to be a string. Non-string inputs will
                  be returned unchanged with a warning log.
            context: A dictionary containing contextual information relevant to the
                     current processing pipeline run. Not directly used by this node,
                     but required by the BaseNode interface.

        Returns:
            The processed data, which is a string with profane words replaced by
            `replacement_char` (repeated for the length of the word), or the
            original data if it's not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"Node '{self.node_name}': Received non-string data of type "
                f"'{type(data).__name__}'. Skipping profanity filtering for this input."
            )
            return data

        processed_text = data
        original_text_hash = hash(data) # Calculate hash to detect if any change occurred

        for pattern in self._compiled_patterns:
            # Define a replacer function that replaces the matched profanity
            # with the `replacement_char` repeated for the length of the matched word.
            def replacer(match):
                return self._replacement_char * len(match.group(0))
            
            processed_text = pattern.sub(replacer, processed_text)
        
        if hash(processed_text) != original_text_hash:
            logger.info(f"Node '{self.node_name}': Profanity detected and successfully filtered.")
        else:
            logger.debug(f"Node '{self.node_name}': No profanity found in the input data.")

        return processed_text