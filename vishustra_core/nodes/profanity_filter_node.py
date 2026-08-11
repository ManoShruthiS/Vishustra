import logging
from typing import Any, Dict, List, Union, Optional

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanity words
    from string data. It supports filtering single strings or lists of strings.
    """

    # A default, basic set of profanity words for demonstration purposes.
    # In a real-world scenario, this might be loaded from a configuration file
    # or a dedicated profanity dictionary service.
    _DEFAULT_PROFANITIES: set[str] = {
        "asshole", "bitch", "cock", "cunt", "damn", "dick", "fuck", "goddamn",
        "hell", "motherfucker", "piss", "shit", "bastard", "wanker"
    }
    _DEFAULT_REPLACEMENT_CHAR: str = '*'

    def __init__(
        self,
        profanities: Optional[List[str]] = None,
        replacement_char: str = _DEFAULT_REPLACEMENT_CHAR
    ) -> None:
        """
        Initializes the ProfanityFilterNode with an optional custom list of profanities
        and a replacement character.

        Args:
            profanities: An optional list of profanity words (case-insensitive) to filter.
                         If None, a default list will be used.
            replacement_char: The character used to replace each filtered profanity.
                              Defaults to '*'.
        """
        self._profanities: set[str] = {word.lower() for word in profanities} if profanities is not None else self._DEFAULT_PROFANITIES
        
        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            logger.warning(
                f"Invalid replacement_char '{replacement_char}' provided. "
                f"Falling back to default '{self._DEFAULT_REPLACEMENT_CHAR}'."
            )
            self._replacement_char = self._DEFAULT_REPLACEMENT_CHAR
        else:
            self._replacement_char = replacement_char
            
        logger.debug(
            f"ProfanityFilterNode initialized with {len(self._profanities)} profanities "
            f"and replacement char: '{self._replacement_char}'"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def _filter_single_string(self, text: str) -> str:
        """
        Filters profanity from a single string.
        This implementation performs a simple whole-word replacement.
        For more advanced filtering (e.g., partial word matches, handling punctuation,
        or leetspeak), a regex-based approach or external library would be necessary.
        """
        if not text:
            return text

        # Split text by whitespace, preserving potential original word structures
        words = text.split()
        filtered_words = []

        for word in words:
            # Remove common leading/trailing punctuation for a more accurate word match
            cleaned_word = word.strip(".,!?;:\"'()[]{}/\\").lower()
            
            if cleaned_word in self._profanities:
                # Replace the original word (or its cleaned part) with replacement characters
                # The length of the replacement matches the original word (or profanity part)
                # for minimal disruption to text flow.
                replacement_length = len(word) 
                
                # A more nuanced replacement would target just the profanity within the word,
                # but for simplicity, we replace the whole detected 'word'.
                # For example, "damn!" -> "****!" or "****". Current impl replaces "damn!" with "****!".
                filtered_words.append(self._replacement_char * replacement_length)
                logger.debug(f"Filtered profanity detected: '{word}'")
            else:
                filtered_words.append(word)
        
        return " ".join(filtered_words)

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter profanities.

        This node accepts:
        - A single string: Filters profanities from the string.
        - A list of strings: Filters profanities from each string in the list.

        For any other data type, a warning is logged, and the original data is returned untouched.
        The `context` dictionary is currently not utilized by this node.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information relevant to the processing flow.

        Returns:
            The processed data with profanities filtered (if applicable), or the original data
            if its type is not supported for filtering.
        """
        if isinstance(data, str):
            return self._filter_single_string(data)
        elif isinstance(data, list):
            # Check if all elements in the list are strings before processing
            if all(isinstance(item, str) for item in data):
                return [self._filter_single_string(item) for item in data]
            else:
                logger.warning(
                    f"ProfanityFilterNode received a list containing non-string items. "
                    f"Cannot filter; returning original data. Data type: {type(data)}."
                )
                return data
        else:
            logger.warning(
                f"ProfanityFilterNode received unsupported data type for filtering. "
                f"Expected str or list[str], but got {type(data)}. Returning original data."
            )
            return data
