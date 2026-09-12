
import logging
import re
from typing import Any, Dict, List, Set

# Assuming the project structure places BaseNode in vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out specified profane words from input strings.
    
    This node identifies and replaces profane words with a configurable mask (e.g., '***').
    The filtering mechanism is case-insensitive and aims to match whole words while
    preserving the original structure and delimiters of the input string.
    """

    def __init__(self, profanity_list: List[str] = None, replacement_mask: str = "***"):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list (List[str], optional): A list of words to be considered profane.
                                                 If None, a predefined default list will be used.
                                                 All words are converted to lowercase internally
                                                 for case-insensitive matching.
            replacement_mask (str, optional): The string used to replace identified profanities.
                                              Defaults to '***'.
        """
        super().__init__()
        # Store profanities in a set for efficient O(1) average-case lookup
        self._profanity_list: Set[str] = {word.lower() for word in (profanity_list if profanity_list else self._default_profanity_list())}
        self._replacement_mask: str = replacement_mask
        logger.debug(f"[{self.node_name}] Initialized with {len(self._profanity_list)} profanities.")

    def _default_profanity_list(self) -> List[str]:
        """
        Provides a default list of common profanities.

        This list is illustrative and can be expanded or loaded dynamically
        from external sources (e.g., a configuration file, a database, or
        a remote service) in a production environment for greater flexibility.
        """
        return ["fuck", "shit", "bitch", "asshole", "cunt", "damn", "cock", "pussy", "bastard"]

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profane words.

        If the input `data` is a string, this method tokenizes it and replaces
        any words found in the configured profanity list with the `replacement_mask`.
        The filtering is case-insensitive, and efforts are made to match whole words.

        If `data` is not a string, a warning is logged, and the data is returned
        without any modification.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        for profanity filtering to occur.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This parameter
                                       is currently unused by this node but is available
                                       for future extensions or global configurations.

        Returns:
            Any: The processed data (a string with profanities filtered) or
                 the original data if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data (type: {type(data).__name__}). "
                "Profanity filtering can only be applied to strings. Returning data as-is."
            )
            return data

        processed_parts: List[str] = []
        # Use regex to split the string into "word" and "non-word" parts.
        # The capturing group `(\W+)` ensures that the delimiters (like spaces,
        # punctuation) are also included in the result list, allowing us to
        # reassemble the string correctly while preserving original formatting.
        parts = re.split(r'(\W+)', data)

        for part in parts:
            # Check if the stripped and lowercased part (which should be a word)
            # is present in our profanity list.
            if part.strip().lower() in self._profanity_list:
                processed_parts.append(self._replacement_mask)
                logger.debug(f"[{self.node_name}] Replaced '{part.strip()}' with mask.")
            else:
                processed_parts.append(part)
        
        # Rejoin all parts to form the filtered string
        filtered_string = "".join(processed_parts)
        logger.info(f"[{self.node_name}] Successfully processed input string for profanity filtering.")
        
        return filtered_string

