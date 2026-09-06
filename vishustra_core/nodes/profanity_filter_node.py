import logging
import re
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the module path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out profane words from text data.
    It replaces identified profane words with a configurable replacement string.
    """

    def __init__(self, profane_words: List[str] = None, replacement: str = '***'):
        """
        Initializes the ProfanityFilterNode with a list of words to filter
        and a replacement string.

        Args:
            profane_words (List[str], optional): A list of words to be considered profane.
                                                 If None, a default list is used.
                                                 Words are processed case-insensitively.
            replacement (str): The string to replace profane words with. Defaults to '***'.
        """
        self._profane_words = [word.lower() for word in profane_words] if profane_words else self._default_profane_words()
        self._replacement = replacement
        logger.debug(f"ProfanityFilterNode initialized with {len(self._profane_words)} profane words and replacement '{self._replacement}'.")

    def _default_profane_words(self) -> List[str]:
        """
        Provides a default list of commonly recognized profane words.
        In a production environment, this list would typically be loaded
        from a secure configuration source or an external service.
        """
        return [
            "ass", "bitch", "bastard", "damn", "dick", "fuck", "shit",
            "cunt", "pussy", "nigger", "faggot", "motherfucker", "cock",
            "bollocks", "wanker", "prick", "whore", "slut"
        ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profane words from the input text data.

        The method expects the input `data` to be a string. It iterates through
        the configured list of profane words and replaces any matches (case-insensitive,
        whole word matches) with the specified replacement string.

        Args:
            data (Any): The input data to be processed, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.

        Returns:
            Any: The filtered string. If the input `data` is not a string,
                 a `TypeError` is raised.

        Raises:
            TypeError: If the input `data` is not an instance of `str`.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name} received unsupported data type: "
                f"'{type(data).__name__}'. Expected 'str'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.debug(f"{self.node_name} starting to filter data (first 50 chars): '{data[:50]}...'")

        filtered_text = data
        for word in self._profane_words:
            # Construct a regex pattern for whole word, case-insensitive matching.
            # re.escape is used to handle special characters within profane words safely.
            pattern = r'\b' + re.escape(word) + r'\b'
            filtered_text = re.sub(pattern, self._replacement, filtered_text, flags=re.IGNORECASE)
        
        logger.debug(f"{self.node_name} finished filtering. Result (first 50 chars): '{filtered_text[:50]}...'")
        return filtered_text