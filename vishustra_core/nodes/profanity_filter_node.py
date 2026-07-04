import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter profanity from text data.

    This node provides functionality to identify and replace specified profane words
    within input strings. It supports recursive processing for data contained
    within lists and dictionaries, ensuring comprehensive content moderation
    across various data structures.

    The node can be configured with a custom list of profanity words, a desired
    replacement character, and case sensitivity settings.
    """

    def __init__(self,
                 profanity_list: Union[List[str], None] = None,
                 replacement_char: str = '*',
                 case_sensitive: bool = False):
        """
        Initializes the ProfanityFilterNode with filtering configurations.

        Args:
            profanity_list: An optional list of words to identify as profanity.
                            If None, a curated default list will be used.
            replacement_char: The character used to obscure identified profanity.
                              Each matched profanity word will be replaced by
                              `replacement_char` repeated for the length of the word.
            case_sensitive: If True, the profanity filter will match words
                            with exact case. Defaults to False for broader matching.
        """
        # A foundational default list for common profanities.
        # In a production system, this would likely be loaded from a robust source.
        _default_profanity_words = [
            "fuck", "shit", "asshole", "bitch", "cunt", "damn", "bastard",
            "piss", "motherfucker", "cock", "dick"
        ]

        self._profanity_list: List[str] = [
            word.lower() if not case_sensitive else word
            for word in (profanity_list if profanity_list is not None else _default_profanity_words)
        ]
        self._replacement_char = replacement_char
        self._case_sensitive = case_sensitive

        logger.debug(
            f"ProfanityFilterNode initialized with {len(self._profanity_list)} "
            f"profanity words. Case-sensitive: {self._case_sensitive}."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilterNode"

    def _filter_string(self, text: str) -> str:
        """
        Applies profanity filtering to a single string.
        Utilizes regex for robust whole-word and case-insensitive matching.
        """
        processed_text = text
        flags = 0 if self._case_sensitive else re.IGNORECASE

        for word_to_filter in self._profanity_list:
            # Compile regex pattern for whole word matching, escaping special chars
            # \b ensures word boundaries, preventing partial word matches (e.g., "ass" in "class")
            pattern = r'\b' + re.escape(word_to_filter) + r'\b'
            replacement = self._replacement_char * len(word_to_filter)
            processed_text = re.sub(pattern, replacement, processed_text, flags=flags)

        return processed_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        The method handles various data types:
        - If `data` is a string, it applies the profanity filter directly.
        - If `data` is a list, it recursively processes each element.
        - If `data` is a dictionary, it recursively processes each value.
        - For other data types, a warning is logged, and the data is returned as-is.

        Args:
            data: The input data, which can be a string, list, or dictionary
                  (potentially nested with strings).
            context: A dictionary containing shared context or metadata
                     relevant to the current orchestration run. Not directly
                     used for filtering logic by this node, but available.

        Returns:
            The processed data with profanity filtered, or the original data
            if its type is not supported for filtering.
        """
        logger.debug(f"ProfanityFilterNode received data of type: {type(data).__name__}")

        try:
            if isinstance(data, str):
                return self._filter_string(data)
            elif isinstance(data, list):
                return [self.process(item, context) for item in data]
            elif isinstance(data, dict):
                return {k: self.process(v, context) for k, v in data.items()}
            else:
                logger.warning(
                    f"ProfanityFilterNode received unsupported data type '{type(data).__name__}'. "
                    "Data will be returned as-is without filtering."
                )
                return data
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during profanity filtering in {self.node_name}: {e}",
                exc_info=True
            )
            # In case of an error, it's often safer to return the original data
            # to prevent pipeline breakage, allowing downstream nodes to potentially
            # handle semi-processed or unprocessed data.
            return data