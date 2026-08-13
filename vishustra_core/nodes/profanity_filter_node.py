import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters profane words from input text.
    It replaces detected profane words with a specified replacement character.
    The filter is case-insensitive and respects word boundaries.
    """

    _DEFAULT_PROFANITY_LIST = [
        'fuck', 'shit', 'bitch', 'asshole', 'damn', 'cunt', 'piss', 'bastard', 'motherfucker'
    ]
    _DEFAULT_REPLACEMENT_CHAR = '*'

    def __init__(self, profanity_list: Optional[List[str]] = None, replacement_char: Optional[str] = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list (Optional[List[str]]): A list of words to consider profane.
                                                 If None, a default list is used.
                                                 Words are internally converted to lowercase.
            replacement_char (Optional[str]): The single character used to replace profane words.
                                            If None, a default '*' is used. If not a single character,
                                            a warning is logged and the default is used.
        """
        # Validate and set replacement character
        self._replacement_char = replacement_char if replacement_char is not None else self._DEFAULT_REPLACEMENT_CHAR
        if not isinstance(self._replacement_char, str) or len(self._replacement_char) != 1:
            logger.warning(
                f"[{self.node_name}] Invalid replacement_char '{self._replacement_char}'. "
                f"Expected a single character string. Falling back to default '{self._DEFAULT_REPLACEMENT_CHAR}'."
            )
            self._replacement_char = self._DEFAULT_REPLACEMENT_CHAR

        # Validate and set profanity list
        processed_profanity_list: List[str] = []
        if profanity_list is not None:
            if not isinstance(profanity_list, list):
                logger.warning(
                    f"[{self.node_name}] Invalid type for profanity_list. Expected 'List[str]', got '{type(profanity_list).__name__}'. "
                    f"Falling back to default profanity list."
                )
                processed_profanity_list = [word.lower() for word in self._DEFAULT_PROFANITY_LIST]
            else:
                for item in profanity_list:
                    if isinstance(item, str):
                        processed_profanity_list.append(item.lower())
                    else:
                        logger.warning(f"[{self.node_name}] Non-string item '{item}' found in profanity_list and ignored.")
        else:
            processed_profanity_list = [word.lower() for word in self._DEFAULT_PROFANITY_LIST]
        
        self._profanity_list = processed_profanity_list
            
        # Compile a regular expression for efficient and robust profanity detection.
        # It handles whole words and is case-insensitive.
        if self._profanity_list:
            # Escape each word to handle special regex characters in profanity list, then join with '|' for OR logic.
            # \b ensures whole word matching.
            self._profanity_regex = re.compile(
                r'\b(' + '|'.join(re.escape(word) for word in self._profanity_list) + r')\b',
                re.IGNORECASE
            )
        else:
            self._profanity_regex = None # No profanity to filter if list is empty

        logger.debug(
            f"[{self.node_name}] Initialized with replacement char: '{self._replacement_char}' "
            f"and profanity list: {self._profanity_list}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profane words.

        Args:
            data (Any): The input data. Expected to be a string that needs filtering.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by this node,
                                       but part of the BaseNode interface.

        Returns:
            Any: The filtered string if the input 'data' was a string and profanity was
                 detected and replaced. Returns the original data if it's not a string
                 or if no profanity was found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type. Expected 'str', got '{type(data).__name__}'. "
                "Returning original data without processing."
            )
            return data

        if not self._profanity_regex:
            logger.debug(f"[{self.node_name}] Profanity list is empty or invalid. No filtering performed.")
            return data

        original_text = data
        
        # Use the compiled regex to find and replace profane words.
        # The lambda function ensures that each matched profane word is replaced
        # with a string of replacement characters of the same length as the original word.
        filtered_text = self._profanity_regex.sub(
            lambda match: self._replacement_char * len(match.group(0)),
            original_text
        )

        if original_text != filtered_text:
            logger.info(f"[{self.node_name}] Successfully filtered profanity from text.")
            logger.debug(f"[{self.node_name}] Original: '{original_text}', Filtered: '{filtered_text}'")
        else:
            logger.debug(f"[{self.node_name}] No profanity found in text.")

        return filtered_text
