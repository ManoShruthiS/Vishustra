import logging
import re
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanity words
    from string data, replacing them with asterisks ('***').

    This node is designed to handle various common data structures containing
    strings, ensuring content moderation at different stages of the pipeline.
    """

    # A simple, illustrative set of profanity words.
    # In a production environment, this list would typically be externalized
    # (e.g., loaded from a configuration file, database, or managed by a
    # dedicated profanity detection service) and potentially much more extensive.
    PROFANITY_WORDS = {
        "fuck", "shit", "damn", "hell", "crap", "bitch", "asshole",
        "bastard", "cock", "dick", "pussy", "tits", "wank", "bollocks",
        "motherfucker", "fucker", "arse", "prick", "slut", "whore"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def _filter_string(self, text: str) -> str:
        """
        Helper method to filter profanity from a single string.

        Args:
            text: The input string to filter.

        Returns:
            The filtered string with profanity replaced by '***'.
        """
        filtered_text = text
        for word in self.PROFANITY_WORDS:
            # Construct a regex pattern for whole word matching, case-insensitive.
            # re.escape() is used to handle special characters in profanity words.
            # \b ensures word boundaries, preventing partial word replacements.
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Perform replacement. re.sub returns the new string.
            new_text = re.sub(pattern, '***', filtered_text, flags=re.IGNORECASE)
            
            if new_text != filtered_text:
                logger.debug(f"[{self.node_name}] Replaced '{word}' in text.")
            
            filtered_text = new_text
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out identified profanity words.

        This method supports filtering within:
        - A single string (`str`).
        - A list of strings (`List[str]`).
        - Dictionary values where the value is a string (`Dict[str, str]`).
          Keys themselves are not filtered, nor are non-string values or
          nested complex structures (lists/dicts within dicts).

        Data types not explicitly supported for filtering will be logged
        with a warning and returned without modification.

        Args:
            data: The input data to be processed. Expected to be a string,
                  list of strings, or a dictionary with string values.
            context: A dictionary containing contextual information relevant
                     to the current processing pipeline.

        Returns:
            The processed data with profanity filtered. If the data type is
            unsupported, the original `data` is returned.
        """
        logger.info(f"[{self.node_name}] Initiating profanity filtering for incoming data.")
        
        if isinstance(data, str):
            original_data = data
            processed_data = self._filter_string(data)
            if original_data != processed_data:
                logger.debug(f"[{self.node_name}] Profanity detected and filtered in string data.")
            else:
                logger.debug(f"[{self.node_name}] No profanity found in string data.")
            return processed_data
        
        elif isinstance(data, list):
            processed_list = []
            has_changes = False
            for i, item in enumerate(data):
                if isinstance(item, str):
                    filtered_item = self._filter_string(item)
                    if filtered_item != item:
                        has_changes = True
                        logger.debug(f"[{self.node_name}] Profanity filtered in list item at index {i}.")
                    processed_list.append(filtered_item)
                else:
                    # Non-string items in a list are passed through unchanged
                    processed_list.append(item) 
            if has_changes:
                logger.debug(f"[{self.node_name}] At least one profanity instance filtered across list data.")
            else:
                logger.debug(f"[{self.node_name}] No profanity found in list data.")
            return processed_list
        
        elif isinstance(data, dict):
            processed_dict = {}
            has_changes = False
            for key, value in data.items():
                if isinstance(value, str):
                    filtered_value = self._filter_string(value)
                    if filtered_value != value:
                        has_changes = True
                        logger.debug(f"[{self.node_name}] Profanity filtered for dictionary key '{key}'.")
                    processed_dict[key] = filtered_value
                else:
                    # Non-string values in a dictionary are passed through unchanged
                    processed_dict[key] = value 
            if has_changes:
                logger.debug(f"[{self.node_name}] At least one profanity instance filtered across dictionary values.")
            else:
                logger.debug(f"[{self.node_name}] No profanity found in dictionary values.")
            return processed_dict
        
        else:
            logger.warning(
                f"[{self.node_name}] Received unsupported data type for profanity filtering: "
                f"'{type(data).__name__}'. Data will be returned unchanged."
            )
            return data