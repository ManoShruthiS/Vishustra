from typing import Any, Dict, List
import logging
import re

# Assuming 'vishustra_core.nodes.base_node' is available in the Python path
# For local development or testing, you might need to adjust sys.path or use a mock.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanities from text data.
    It replaces identified profane words or phrases with a configurable replacement character.
    """

    _DEFAULT_PROFANITIES = [
        "badword", "anotherbad", "terriblephrase", "profanityexample"
    ]

    def __init__(self, profanity_list: List[str] = None, replacement_char: str = "*"):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list (List[str], optional): A list of words/phrases to filter.
                                                  If None, a predefined internal list is used.
                                                  The list will be converted to lowercase for case-insensitive matching.
            replacement_char (str, optional): The character to use for replacing profanities.
                                              Defaults to '*'. This character will be repeated
                                              to match the length of the filtered word.
        """
        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            logger.warning(
                f"[{self.node_name}] Invalid replacement_char '{replacement_char}'. "
                "Falling back to default '*'."
            )
            self._replacement_char = "*"
        else:
            self._replacement_char = replacement_char

        effective_profanity_list = profanity_list if profanity_list is not None else self._DEFAULT_PROFANITIES
        if not isinstance(effective_profanity_list, list):
            logger.error(
                f"[{self.node_name}] profanity_list must be a list of strings, but got {type(effective_profanity_list).__name__}. "
                "Initializing with an empty list."
            )
            self._profanity_list = []
        else:
            self._profanity_list = [str(p).lower() for p in effective_profanity_list if isinstance(p, str)]
            if len(self._profanity_list) != len(effective_profanity_list):
                logger.warning(
                    f"[{self.node_name}] Some items in profanity_list were not strings and were skipped."
                )

        logger.info(
            f"[{self.node_name}] Initialized with {len(self._profanity_list)} profanities "
            f"and replacement character '{self._replacement_char}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, replacing identified profanities with the configured
        replacement character, maintaining the original string's overall length.

        Args:
            data (Any): The input data. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing pipeline.

        Returns:
            Any: The processed data (string with profanities filtered), or the original data
                 if it's not a string or if no profanities were found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data).__name__}. "
                "Profanity filtering will be skipped and original data returned."
            )
            return data

        processed_text = data
        found_profanity = False

        for profanity_term in self._profanity_list:
            if not profanity_term: # Skip empty strings in profanity list
                continue

            # Compile a regex pattern for case-insensitive matching of the profanity term.
            # re.escape is used to treat the profanity_term as a literal string in the regex.
            pattern = re.compile(re.escape(profanity_term), re.IGNORECASE)

            # Use re.sub with a replacer function to replace matches
            # The replacer function ensures the replacement length matches the original profanity length.
            def replacer(match):
                nonlocal found_profanity # Indicate that a profanity was found
                found_profanity = True
                return self._replacement_char * len(match.group(0))

            processed_text = pattern.sub(replacer, processed_text)

        if found_profanity:
            logger.info(f"[{self.node_name}] Filtered profanities from data.")
        else:
            logger.debug(f"[{self.node_name}] No profanities found in data.")

        return processed_text