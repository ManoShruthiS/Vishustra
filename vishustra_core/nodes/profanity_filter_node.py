
import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists and contains BaseNode
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node that filters specified profanity from input text.
    It supports string, list of strings, and dictionary (string values) inputs.
    """

    _DEFAULT_BANNED_WORDS = [
        "badword1", "badword2", "damn", "ass", "shit", "fuck", "bitch", 
        "cunt", "prick", "motherfucker", "bastard"
    ]
    _DEFAULT_REPLACEMENT_CHAR = "*"

    def __init__(self, banned_words: List[str] = None, replacement_char: str = None):
        """
        Initializes the ProfanityFilterNode with a custom list of banned words
        and a replacement character, or uses default values.

        Args:
            banned_words (List[str], optional): A list of words to be filtered. 
                                                Defaults to a predefined list.
            replacement_char (str, optional): The character to use for replacing 
                                              filtered words. Defaults to '*'.
        """
        self._banned_words = [word.lower() for word in (banned_words or self._DEFAULT_BANNED_WORDS)]
        
        if not replacement_char:
            self._replacement_char = self._DEFAULT_REPLACEMENT_CHAR
        elif len(replacement_char) != 1:
            logger.warning(
                f"[{self.node_name}] Replacement character must be a single character. "
                f"Received '{replacement_char}'. Falling back to default '{self._DEFAULT_REPLACEMENT_CHAR}'."
            )
            self._replacement_char = self._DEFAULT_REPLACEMENT_CHAR
        else:
            self._replacement_char = replacement_char
            
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._banned_words)} banned words "
            f"and replacement '{self._replacement_char}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def _filter_text(self, text: str) -> str:
        """
        Applies profanity filtering to a single string.
        Replaces each banned word with `replacement_char` repeated 
        to match the length of the original word.
        """
        if not isinstance(text, str):
            logger.warning(
                f"[{self.node_name}] _filter_text received non-string input: {type(text)}. "
                "Returning input as is without filtering."
            )
            return text

        original_text = text
        filtered_text = text
        
        for banned_word in self._banned_words:
            # Create a regex pattern for the banned word, ensuring whole word match (\b)
            # and case insensitivity (re.IGNORECASE). re.escape handles special characters.
            pattern = r'\b' + re.escape(banned_word) + r'\b'
            
            # The replacement string's length should match the banned word's length
            replacement = self._replacement_char * len(banned_word)
            
            # Use re.sub to replace all occurrences
            filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        if original_text != filtered_text:
            logger.debug(
                f"[{self.node_name}] Filtered text. "
                f"Original (first 50 chars): '{original_text[:50]}{'...' if len(original_text) > 50 else ''}', "
                f"Filtered (first 50 chars): '{filtered_text[:50]}{'...' if len(filtered_text) > 50 else ''}'"
            )
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profanity.
        
        Supported input types:
        - str: The string will be filtered directly.
        - List[str]: Each string in the list will be filtered. Non-string elements are passed through.
        - Dict[str, Any]: String values in the dictionary will be filtered. Other values are passed through.

        Args:
            data (Any): The input data to be processed.
            context (Dict[str, Any]): A dictionary containing contextual information 
                                      for the processing pipeline (not used by this node).

        Returns:
            Any: The processed data with profanity filtered.
        """
        logger.info(f"[{self.node_name}] Initiating profanity filtering process.")
        
        try:
            if isinstance(data, str):
                return self._filter_text(data)
            elif isinstance(data, list):
                return [self._filter_text(item) if isinstance(item, str) else item for item in data]
            elif isinstance(data, dict):
                processed_data = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        processed_data[key] = self._filter_text(value)
                    else:
                        processed_data[key] = value
                return processed_data
            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported data type for profanity filtering: {type(data)}. "
                    "Returning data as is without processing."
                )
                return data
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during processing: {e}", 
                exc_info=True
            )
            # Depending on error handling strategy, might re-raise or return original data
            raise # Re-raise to ensure upstream nodes are aware of failure
