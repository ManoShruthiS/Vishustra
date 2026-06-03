import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available in the specified module path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters profanity from text data.
    It replaces specified profane words with a masked version (e.g., '****').
    """

    def __init__(self, profanity_list: List[str] = None, mask_char: str = '*', case_sensitive: bool = False):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list (List[str], optional): A list of words to be considered profane.
                                                 If None, a small predefined list is used.
            mask_char (str, optional): The character to use for masking profane words. Defaults to '*'.
                                       This character will be repeated to match the length of the profane word.
            case_sensitive (bool, optional): Whether the profanity filter should be case-sensitive. Defaults to False.
                                             Note: If `case_sensitive` is False, all provided profanity words will
                                             be converted to lowercase internally for consistent matching.
        """
        if profanity_list is None:
            self._profanity_list = self._default_profanity_list()
        else:
            self._profanity_list = [word if case_sensitive else word.lower() for word in profanity_list]
            
        # Remove duplicates from the profanity list and sort for consistency
        self._profanity_list = sorted(list(set(self._profanity_list)))

        self._mask_char = mask_char
        self._case_sensitive = case_sensitive
        logger.debug(f"[{self.node_name}] Initialized with profanities: {self._profanity_list}, mask_char: '{self._mask_char}', case_sensitive: {self._case_sensitive}")

    def _default_profanity_list(self) -> List[str]:
        """Provides a default list of common profanities if none are supplied."""
        return ["damn", "ass", "shit", "bitch", "fuck", "cunt"]

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profanity from the input text data by replacing detected words with mask characters.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): The context dictionary for processing.

        Returns:
            Any: The processed data with profanity masked.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Received non-string data of type {type(data).__name__}. Expected string.")
            raise TypeError(f"{self.node_name} expects input 'data' to be a string, but received {type(data).__name__}.")

        processed_data = data
        detected_profanities = set() # Use a set to avoid duplicate logging of the same word

        logger.info(f"[{self.node_name}] Starting profanity filtering for input data (first 75 chars): '{data[:75]}{'...' if len(data) > 75 else ''}'")

        for word in self._profanity_list:
            # Create a regex pattern for the word, ensuring whole word matching (\b for word boundary)
            # re.escape handles special characters in profanity list (e.g., 'f*ck')
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE if not self._case_sensitive else 0)
            
            # Use finditer to find all occurrences and log them
            found_matches = False
            for match in pattern.finditer(processed_data):
                detected_profanities.add(match.group(0)) # Log the actual matched string
                found_matches = True
            
            if found_matches:
                replacement = self._mask_char * len(word)
                processed_data = pattern.sub(replacement, processed_data)
                logger.debug(f"[{self.node_name}] Replaced occurrences of '{word}' with '{replacement}'.")

        if detected_profanities:
            logger.warning(f"[{self.node_name}] Profanities detected and masked: {', '.join(sorted(list(detected_profanities)))}")
        else:
            logger.info(f"[{self.node_name}] No profanities detected in the input data.")
            
        return processed_data