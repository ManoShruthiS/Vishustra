import logging
import re
from typing import Any, Dict

# Assuming BaseNode is located in the specified core module
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to detect and filter out common profanity
    from textual data. It replaces identified profane words with a configurable
    censored string.

    This node supports filtering for:
    - Raw string inputs.
    - Dictionary inputs, targeting string values associated with common text keys.

    For other data types, it will log a warning and pass the data through unchanged.
    """

    # A curated list of profane words. In a production environment, this list
    # would typically be externalized (e.g., config file, database, dedicated library)
    # and potentially dynamically updated.
    _default_profane_words = {
        "damn", "hell", "shit", "fuck", "bitch", "asshole", "cunt", "prick",
        "bastard", "motherfucker", "fucker", "cock", "dick", "pussy"
    }

    def __init__(self, replacement_string: str = "***", custom_profane_words: set[str] = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            replacement_string: The string used to replace detected profane words.
                                Defaults to '***'.
            custom_profane_words: An optional set of additional profane words to include.
                                  These will be merged with the node's default list.
        """
        self._replacement_string = replacement_string
        
        # Combine default and custom profane words, ensuring uniqueness and lowercasing
        effective_profane_words = self._default_profane_words.copy()
        if custom_profane_words:
            effective_profane_words.update(word.lower() for word in custom_profane_words)

        # Compile a regular expression for efficient, case-insensitive, whole-word matching.
        # \b ensures word boundaries, and re.escape handles special regex characters.
        self._profanity_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in sorted(list(effective_profane_words))) + r')\b',
            re.IGNORECASE
        )
        logger.debug(
            f"ProfanityFilterNode initialized with replacement: '{replacement_string}' "
            f"and {len(effective_profane_words)} profane words."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilter"

    def _censor_text(self, text: str) -> str:
        """
        Internal method to apply the profanity filter to a given string.
        """
        if not text:
            return text

        original_text = text
        censored_text = self._profanity_pattern.sub(self._replacement_string, text)
        
        if original_text != censored_text:
            logger.info(f"Profanity detected and censored in text. Original snippet: '{original_text[:75]}...'")
        
        return censored_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and censor profanity.

        The method handles different data types:
        - If `data` is a string, it applies the filter directly.
        - If `data` is a dictionary, it iterates through common text-related keys
          (e.g., 'text', 'content', 'message') and filters their string values.
          A copy of the dictionary is returned with updated values.
        - For any other data type, a warning is logged, and the data is returned
          unmodified.

        Args:
            data: The input data to be processed. This can be a string, a dictionary
                  containing string values, or another data type.
            context: A dictionary providing contextual information for the node's operation.
                     This node does not currently use the context directly but adheres
                     to the `BaseNode` interface.

        Returns:
            The processed data with profanity filtered, or the original data if
            filtering was not applicable or the data type was unsupported.
        """
        logger.debug(f"ProfanityFilterNode received data for processing. Data type: {type(data).__name__}")

        if isinstance(data, str):
            return self._censor_text(data)
        
        elif isinstance(data, dict):
            processed_data = data.copy()  # Work on a copy to avoid modifying the original
            text_keys_to_check = ['text', 'content', 'message', 'query', 'response', 'utterance']
            
            modified = False
            for key in text_keys_to_check:
                if key in processed_data and isinstance(processed_data[key], str):
                    original_value = processed_data[key]
                    censored_value = self._censor_text(original_value)
                    if original_value != censored_value:
                        processed_data[key] = censored_value
                        modified = True
            
            if not modified:
                logger.debug(
                    f"ProfanityFilterNode processed dictionary data but found no string values "
                    f"under common text keys {text_keys_to_check} to filter. Data returned unchanged."
                )
            return processed_data
        
        else:
            logger.warning(
                f"ProfanityFilterNode received unsupported data type '{type(data).__name__}'. "
                f"Expected 'str' or 'dict'. Data will be returned unchanged."
            )
            return data