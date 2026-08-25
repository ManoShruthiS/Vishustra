import logging
import re
from typing import Any, Dict, List, Optional

# Assuming BaseNode is located here within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out common profanity from string data.
    It identifies and replaces specified profanity words with a configurable
    masked string (e.g., '***'). The filtering process is case-insensitive
    and respects word boundaries to avoid unintended replacements within words.
    """

    _logger = logging.getLogger(__name__)

    # Default list of profanity words. This can be extended or loaded from
    # a configuration service in a production environment.
    _default_profanity_list: List[str] = [
        "ass", "bitch", "cunt", "damn", "dick", "fuck", "hell", "shit",
        "piss", "slut", "whore", "cock", "motherfucker", "bastard"
    ]
    _default_replacement_string: str = "***"

    def __init__(self, custom_profanity_list: Optional[List[str]] = None, 
                 replacement_string: Optional[str] = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            custom_profanity_list: An optional list of strings to use as profanity.
                                   If None, the default list will be used.
            replacement_string: An optional string to replace profanity with.
                                If None, the default replacement string '***' will be used.
        """
        self._profanity_list: List[str] = custom_profanity_list if custom_profanity_list is not None else self._default_profanity_list
        self._replacement_string: str = replacement_string if replacement_string is not None else self._default_replacement_string
        
        self._logger.debug(
            f"{self.node_name} initialized with {len(self._profanity_list)} profanity words "
            f"and replacement string: '{self._replacement_string}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profanity from the input data.

        If the input `data` is a string, it iteratively replaces occurrences of
        words from the internal profanity list with the configured replacement string.
        The matching is case-insensitive and targets whole words.

        If `data` is not a string, a warning is logged, and the data is returned unchanged.
        Context information is acknowledged but not directly utilized for filtering logic.

        Args:
            data: The input data to be processed, typically expected to be a string.
            context: A dictionary containing contextual information relevant to the
                     orchestration flow. This node does not modify its behavior based on context.

        Returns:
            The processed data (a string with profanity filtered out) or the original
            data if it was not a string or encountered an error.
        """
        if context:
            self._logger.debug(
                f"{self.node_name} received context with keys: {list(context.keys())}, "
                "but does not utilize it for profanity filtering."
            )

        if not isinstance(data, str):
            self._logger.warning(
                f"{self.node_name} received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering can only be applied to strings. Returning data unchanged."
            )
            return data

        filtered_text = str(data)  # Work with a mutable copy of the string

        for word in self._profanity_list:
            # Construct a regex pattern for whole word matching (\b) and case-insensitivity (re.IGNORECASE).
            # re.escape() is used to ensure that special regex characters within 'word' are treated literally.
            pattern = r'\b' + re.escape(word) + r'\b'
            try:
                filtered_text = re.sub(pattern, self._replacement_string, filtered_text, flags=re.IGNORECASE)
                self._logger.debug(
                    f"Successfully replaced instances of '{word}' with '{self._replacement_string}'."
                )
            except re.error as e:
                self._logger.error(
                    f"Regex error while filtering word '{word}' in {self.node_name}: {e}. "
                    "Skipping this word."
                )
            except Exception as e:
                self._logger.error(
                    f"An unexpected error occurred while filtering word '{word}' in {self.node_name}: {e}. "
                    "Skipping this word."
                )

        self._logger.info(f"{self.node_name} successfully processed data, filtering profanity.")
        return filtered_text