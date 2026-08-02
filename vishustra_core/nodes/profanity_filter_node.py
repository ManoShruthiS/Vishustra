import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node for Vishustra that filters out common profanity from text.
    It replaces detected profanity with asterisks ('***').
    
    This node is designed to handle single strings or lists of strings,
    providing a basic content moderation capability within the orchestration framework.
    """

    # This list can be expanded significantly, loaded from a configuration file,
    # or an external service in a production environment.
    _PROFANITY_WORDS = [
        "ass", "bastard", "bitch", "cunt", "damn", "dick", "fag", "fuck",
        "hell", "motherfucker", "nigga", "nigger", "piss", "pussy", "shit",
        "slut", "whore", "bollocks", "arse"
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def _filter_text(self, text: str) -> str:
        """
        Helper method to filter profanity from a single string.
        Performs case-insensitive replacement of whole words using regular expressions.
        """
        filtered_text = text
        for word in self._PROFANITY_WORDS:
            # Construct a regex pattern for the profane word.
            # \b ensures whole word matching (e.g., 'cat' won't match 'catamaran').
            # re.escape handles words that might contain special regex characters (e.g., "f.u.c.k").
            # re.IGNORECASE makes the matching case-insensitive.
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Keep track of the text before substitution to check if a change occurred.
            original_text_segment = filtered_text
            
            # Replace all occurrences of the profane word with '***'.
            filtered_text = re.sub(pattern, '***', filtered_text, flags=re.IGNORECASE)
            
            # Log only if an actual replacement happened for this word.
            if original_text_segment != filtered_text:
                logger.debug(f"[{self.node_name}] Replaced occurrences of '{word}' in the text segment.")
                
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter profanity.
        
        The method expects input data to be either a single string or a list of strings.
        Each string is then passed through the profanity filter.
        
        Args:
            data (Any): The input data to be processed. Expected types are `str` or `List[str]`.
            context (Dict[str, Any]): A dictionary containing contextual information 
                                       relevant to the current execution flow.
        
        Returns:
            Any: The processed data with profanity filtered. 
                 Returns `str` if input was `str`, or `List[str]` if input was `List[str]`.
        
        Raises:
            TypeError: If the input data is not a string or a list of strings, 
                       an error is logged and a `TypeError` is raised.
        """
        logger.info(f"[{self.node_name}] Starting profanity filter process.")
        logger.debug(f"[{self.node_name}] Input data type detected: {type(data).__name__}.")

        if isinstance(data, str):
            filtered_data = self._filter_text(data)
            logger.info(f"[{self.node_name}] Profanity filtering completed for a single string.")
            return filtered_data
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            # Process each string in the list
            filtered_list = [self._filter_text(item) for item in data]
            logger.info(f"[{self.node_name}] Profanity filtering completed for a list of strings.")
            return filtered_list
        else:
            # Handle unsupported data types gracefully by logging and raising an error
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)