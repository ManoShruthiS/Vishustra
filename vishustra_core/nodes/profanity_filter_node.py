import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters profanity from text data.

    This node replaces specified profane words within the input text
    with a chosen replacement character (e.g., asterisks). It handles
    case-insensitivity and ensures whole word matching, replacing each
    detected profane word with a string of replacement characters of
    the same length.
    """

    def __init__(
        self,
        profanity_list: Optional[List[str]] = None,
        replacement_char: str = '*',
    ):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list: An optional list of words to be considered profane.
                            If None, a minimal default list is used for demonstration.
                            Users are encouraged to provide a comprehensive list.
            replacement_char: The character used to replace profane words. Defaults to '*'.
        """
        super().__init__()
        # Ensure profanity words are lowercased for consistent internal processing.
        # A small default list is provided for illustrative purposes.
        self._profanity_list = [word.lower() for word in profanity_list] if profanity_list else [
            "fuck", "shit", "bitch", "asshole", "cunt", "damn"
        ]
        self._replacement_char = replacement_char

        # Compile regex patterns for efficient and case-insensitive whole-word matching.
        # The list is sorted by word length (descending) and then alphabetically.
        # This strategy helps prevent partial matches where a shorter profanity word
        # is a substring of a longer one (e.g., ensuring "bullshit" is matched before "shit").
        self._profanity_patterns = []
        sorted_profanity = sorted(self._profanity_list, key=lambda x: (-len(x), x))

        for word in sorted_profanity:
            # Use '\b' for word boundaries to ensure whole word matching (e.g., "shit" but not "spreadsheet").
            # re.escape is crucial to treat the profanity word literally, preventing issues
            # if a word contains characters that are special in regex (e.g., "f.ck").
            pattern = r'\b' + re.escape(word) + r'\b'
            self._profanity_patterns.append(re.compile(pattern, re.IGNORECASE))
        
        logger.debug(
            f"ProfanityFilterNode initialized with {len(self._profanity_list)} profanity words "
            f"and replacement char: '{self._replacement_char}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        The node expects 'data' to be a string. If 'data' is not a string,
        it will be converted to a string before processing if possible.

        Args:
            data: The input data, expected to be a string containing text to be filtered.
            context: A dictionary containing contextual information for the processing.
                     (Not directly used by this node, but part of the BaseNode interface).

        Returns:
            The processed string with profanity filtered, or the original data
            (or its string representation) if it could not be processed due to errors.
        """
        processed_data = data

        # Attempt to convert non-string data to string for processing.
        if not isinstance(data, str):
            try:
                processed_data = str(data)
                logger.warning(
                    f"ProfanityFilterNode received non-string data (type: {type(data).__name__}). "
                    "Attempting to convert to string for processing."
                )
            except Exception as e:
                logger.error(
                    f"ProfanityFilterNode failed to convert data of type {type(data).__name__} "
                    f"to string for processing: {e}", exc_info=True
                )
                return data # Return original data if conversion to string fails

        original_text_to_compare = processed_data
        
        try:
            for pattern in self._profanity_patterns:
                # Use a lambda function for the replacement to dynamically match
                # the length of the detected profane word with the replacement characters.
                processed_data = pattern.sub(
                    lambda m: self._replacement_char * len(m.group(0)),
                    processed_data
                )
            
            if original_text_to_compare != processed_data:
                logger.info("Profanity detected and filtered from text.")
            else:
                logger.debug("No profanity detected in text.")
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error during profanity filtering process: {e}", exc_info=True)
            # In case of an unexpected error during regex processing,
            # return the original (potentially string-converted) data.
            return original_text_to_compare