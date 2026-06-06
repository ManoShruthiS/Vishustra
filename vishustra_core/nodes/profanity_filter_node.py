import logging
import re
from typing import Any, Dict, List, Optional

# Vishustra-specific import for the base node.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters specified profanity words from input text.

    It replaces identified profanity words with a configurable replacement character,
    matching words case-insensitively and respecting word boundaries to avoid
    replacing parts of non-profane words (e.g., 'passage' vs. 'ass').
    The length of the original profanity word is maintained by repeating the
    replacement character.
    """

    def __init__(self, profanity_list: Optional[List[str]] = None, replacement_char: str = '*'):
        """
        Initializes the ProfanityFilterNode with a list of profanity words
        and a character to use for replacement.

        Args:
            profanity_list (Optional[List[str]]): A list of words to filter out.
                                                 If None, a sensible default list of common
                                                 profanities is used. Words will be matched
                                                 case-insensitively.
            replacement_char (str): The single character to replace profanity words with.
                                  Defaults to '*'. If an invalid character (not a single
                                  character string) is provided, it defaults to '*'.
        """
        if profanity_list is None:
            self._profanity_list = [
                "fuck", "shit", "asshole", "bitch", "cunt", "damn", "piss", "motherfucker"
            ]
            logger.debug("ProfanityFilterNode initialized with default profanity list.")
        else:
            self._profanity_list = profanity_list
            logger.debug(f"ProfanityFilterNode initialized with custom profanity list: {self._profanity_list}")

        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            logger.warning(
                f"Invalid 'replacement_char' received: '{replacement_char}'. Must be a single character string. "
                "Defaulting to '*' for profanity replacement."
            )
            self._replacement_char = '*'
        else:
            self._replacement_char = replacement_char
        
        logger.info(
            f"ProfanityFilterNode configuration: {len(self._profanity_list)} profanity words loaded, "
            f"replacement character: '{self._replacement_char}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profanity from the input data (expected to be a string) by replacing
        identified words with the specified replacement character.

        Matches are case-insensitive and respect word boundaries (`\b` in regex)
        to prevent partial word replacements (e.g., 'passage' remains 'passage',
        not 'pa***ge'). The length of the original profanity word is maintained
        by repeating the replacement character.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        or an object convertible to a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                    for the processing flow (not explicitly used by this node,
                                    but passed along Vishustra convention).

        Returns:
            Any: The processed data with profanity filtered. If the input `data`
                 was not a string, the original `data` is returned unchanged,
                 with a warning logged.
        """
        if not isinstance(data, str):
            logger.warning(
                f"ProfanityFilterNode received non-string input data (type: {type(data)}). "
                "Returning original data without filtering as no text processing can be performed."
            )
            return data

        processed_text = data
        original_length = len(data)

        for word in self._profanity_list:
            # Construct a regex pattern for whole words, case-insensitively.
            # re.escape() ensures that if a profanity word contains regex special characters,
            # they are treated literally.
            # \b matches word boundaries, preventing partial matches like 'ass' in 'passage'.
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            
            # Use re.sub with a lambda function as the replacement.
            # The lambda receives a match object. It replaces the matched string (m.group(0))
            # with the replacement character repeated for the length of the original match,
            # ensuring consistent string length before and after filtering.
            processed_text = pattern.sub(
                lambda m: self._replacement_char * len(m.group(0)),
                processed_text
            )

        logger.debug(f"ProfanityFilterNode processed data. Original text length: {original_length}, "
                     f"Filtered text length: {len(processed_text)}.")
        return processed_text