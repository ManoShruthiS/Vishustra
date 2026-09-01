from typing import Any, Dict, List
import logging
import re

# CRITICAL: This import path is specified by the project context.
# Assuming 'vishustra_core.nodes.base_node' resolves to the BaseNode class provided
# in the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profane words
    from string data, replacing them with a custom string.

    This node is designed to sanitize text inputs by identifying and
    replacing offensive language with a user-defined placeholder.
    """

    def __init__(self, profane_words: List[str] = None, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode with a list of words to filter
        and a replacement string.

        Args:
            profane_words (List[str], optional): A list of words (case-insensitive)
                                                  to be identified as profane. If None,
                                                  a predefined default list is used.
            replacement_string (str, optional): The string to substitute for each
                                                identified profane word. Defaults to "***".
        """
        # Convert all profane words to lowercase for consistent, case-insensitive matching
        self._profane_words = [word.lower() for word in profane_words] if profane_words else self._default_profane_words()
        self._replacement_string = replacement_string
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._profane_words)} profane words "
            f"and replacement string: '{self._replacement_string}'"
        )

    def _default_profane_words(self) -> List[str]:
        """
        Provides a default list of common profane words.
        In a production environment, this list would likely be loaded
        from a configuration file or a database.
        """
        return ["damn", "hell", "shit", "fuck", "bitch", "asshole", "cunt", "bastard"]

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profane words.

        If the input `data` is not a string, a warning is logged, and the
        original data is returned unchanged. For string inputs, the method
        iterates through the configured profane words, replacing each occurrence
        (case-insensitively and respecting word boundaries) with the defined
        `replacement_string`.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        for filtering to occur.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing pipeline. This node does
                                       not currently make use of the context.

        Returns:
            Any: The filtered string data if `data` was a string, otherwise the
                 original `data` unmodified.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering is applicable only to strings. Returning data unchanged."
            )
            return data

        processed_data = data
        for word in self._profane_words:
            # Construct a regex pattern for whole-word, case-insensitive matching.
            # re.escape() is used to handle special characters within the profane words,
            # and '\b' ensures that only whole words are matched (e.g., 'hell' matches
            # "what the hell" but not "hello").
            pattern = r'\b' + re.escape(word) + r'\b'
            processed_data = re.sub(pattern, self._replacement_string, processed_data, flags=re.IGNORECASE)
        
        logger.debug(f"[{self.node_name}] Successfully filtered data.")
        return processed_data