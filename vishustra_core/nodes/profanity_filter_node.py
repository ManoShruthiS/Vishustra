
import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path in the Vishustra core
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanity words
    from text data, replacing them with a configurable character sequence.
    """

    # Default list of profanity words to filter.
    # In a production environment, this might be loaded from a configuration
    # file, a database, or an external service.
    DEFAULT_PROFANITY_LIST = ["fuck", "shit", "bitch", "asshole", "damn", "cunt", "pussy", "dick"]
    
    # Default character sequence used to replace profanity.
    DEFAULT_REPLACEMENT_CHAR = "***"

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by identifying and replacing profanity.

        The node expects 'data' to be a string. If 'data' is not a string,
        a warning is logged, and the original data is returned unmodified.

        Context parameters:
        - 'profanity_list' (list[str], optional): A custom list of words
          to treat as profanity. Overrides `DEFAULT_PROFANITY_LIST`.
        - 'replacement_char' (str, optional): The string to use as a
          replacement for detected profanity. Overrides `DEFAULT_REPLACEMENT_CHAR`.

        Args:
            data: The input data, expected to be a string containing text.
            context: A dictionary containing operational context, which may
                     include configuration for filtering.

        Returns:
            The processed string with profanity filtered, or the original
            data if it was not a string or an error occurred during processing.
        """
        logger.debug(f"[{self.node_name}] Initiating profanity filtering process.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering is applicable only to strings. Returning original data."
            )
            return data

        # Determine the profanity list to use
        profanity_list = context.get(
            "profanity_list", self.DEFAULT_PROFANITY_LIST
        )
        if not isinstance(profanity_list, list) or not all(isinstance(w, str) for w in profanity_list):
            logger.error(
                f"[{self.node_name}] Invalid 'profanity_list' provided in context. "
                f"Expected list of strings, got {type(profanity_list).__name__}. "
                "Falling back to default profanity list."
            )
            profanity_list = self.DEFAULT_PROFANITY_LIST
        
        if not profanity_list: # Ensure list is not empty to avoid regex errors
            logger.debug(f"[{self.node_name}] Profanity list is empty. No filtering will be applied.")
            return data

        # Determine the replacement character
        replacement_char = context.get(
            "replacement_char", self.DEFAULT_REPLACEMENT_CHAR
        )
        if not isinstance(replacement_char, str):
            logger.error(
                f"[{self.node_name}] Invalid 'replacement_char' provided in context. "
                f"Expected string, got {type(replacement_char).__name__}. "
                "Falling back to default replacement character."
            )
            replacement_char = self.DEFAULT_REPLACEMENT_CHAR

        try:
            # Construct a regex pattern for case-insensitive whole-word matching.
            # `re.escape` handles special characters in profanity words.
            # `\b` ensures whole word matching (e.g., "fuck" but not "firetruck").
            # `(?:...)` is a non-capturing group.
            profanity_pattern = re.compile(
                r'\b(?:' + '|'.join(re.escape(word) for word in profanity_list) + r')\b',
                re.IGNORECASE
            )
            
            # Apply the replacement
            filtered_text = profanity_pattern.sub(replacement_char, data)
            
            logger.debug(f"[{self.node_name}] Successfully filtered profanity.")
            return filtered_text

        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during profanity filtering: {e}. "
                "Returning original data to prevent data loss."
            )
            return data

