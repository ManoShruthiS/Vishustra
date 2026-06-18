import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is located in the specified path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out common profane words from text data.
    It replaces identified profanities with a configurable mask (e.g., '***')
    to ensure text output is appropriate for various contexts.
    """

    # A set of common profane words for filtering. This list can be extended
    # or loaded from a configuration file in a production environment.
    # Words are matched case-insensitively.
    _PROFANITIES: set[str] = {
        "ass", "bitch", "cunt", "damn", "fuck", "hell", "piss", "shit", "bastard",
        "cock", "motherfucker", "fucker", "tits", "arse", "bollocks", "wanker",
        "prick", "whore", "slut"
    }
    # The string used to mask profane words.
    _REPLACEMENT_MASK: str = "***"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def _filter_text(self, text: str) -> str:
        """
        Helper method to perform profanity filtering on a single string.
        It uses regular expressions to ensure case-insensitive, whole-word matching.

        Args:
            text (str): The input string to be filtered.

        Returns:
            str: The filtered string with profanities replaced by the mask.
        """
        if not isinstance(text, str):
            logger.warning(
                "[%s] _filter_text received non-string input (type: %s). Returning original data.",
                self.node_name, type(text).__name__
            )
            # Depending on strictness, an error could be raised here instead.
            return text

        processed_text = text
        for word in self._PROFANITIES:
            # Construct a regex pattern for whole-word matching (\b) and case-insensitivity.
            # re.escape is used to handle special characters in profane words safely.
            pattern = r'\b' + re.escape(word) + r'\b'
            processed_text = re.sub(pattern, self._REPLACEMENT_MASK, processed_text, flags=re.IGNORECASE)
            # logger.debug("[%s] Replaced instances of '%s'.", self.node_name, word) # Too verbose for a loop

        return processed_text

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[str, List[str]]:
        """
        Processes the input data, applying profanity filtering.
        This method supports filtering either a single string or a list of strings.

        Args:
            data (Union[str, List[str]]): The input data. Expected to be a string
                                          or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing. This node does not directly
                                      utilize the context but it's available for
                                      future extensions or orchestration.

        Returns:
            Union[str, List[str]]: The processed data with profane words masked.
                                   If the input was a string, a string is returned.
                                   If the input was a list, a list is returned.

        Raises:
            TypeError: If the input 'data' is None or not of type `str` or `List[str]`.
        """
        logger.info("[%s] Initiating profanity filtering process.", self.node_name)
        logger.debug("[%s] Input data type received: %s", self.node_name, type(data).__name__)

        if data is None:
            logger.error("[%s] Received None as input data. Processing aborted.", self.node_name)
            raise TypeError(f"[{self.node_name}] Input 'data' cannot be None.")

        if isinstance(data, str):
            filtered_data = self._filter_text(data)
            logger.debug("[%s] Successfully filtered single string data.", self.node_name)
            return filtered_data
        elif isinstance(data, list):
            # Check for non-string elements within the list for robust handling.
            if not all(isinstance(item, str) for item in data):
                logger.warning(
                    "[%s] Input list contains non-string elements. Attempting to filter string elements only "
                    "and preserving non-string elements as-is.",
                    self.node_name
                )
                filtered_list = [self._filter_text(item) if isinstance(item, str) else item for item in data]
            else:
                filtered_list = [self._filter_text(item) for item in data]
            logger.debug("[%s] Successfully filtered list of strings data.", self.node_name)
            return filtered_list
        else:
            logger.error(
                "[%s] Invalid input data type: %s. Expected 'str' or 'List[str]'.",
                self.node_name, type(data).__name__
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string or a list of strings, "
                f"but received type '{type(data).__name__}'."
            )