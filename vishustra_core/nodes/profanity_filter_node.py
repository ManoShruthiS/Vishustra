import logging
import re
from typing import Any, Dict

# Simulating the import path for BaseNode
# In a real project, this would be:
# from vishustra_core.nodes.base_node import BaseNode
from abc import ABC, abstractmethod

class BaseNode(ABC):
    """
    Base class for all Vishustra processing nodes.
    Each node must implement the process method.
    """
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data and returns the result.
        """
        pass
        
    @property
    @abstractmethod
    def node_name(self) -> str:
        """Returns the name of the node."""
        pass


logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanities
    from string data, replacing them with asterisks.
    """

    # A simple, configurable list of profane words.
    # In a production system, this would likely be loaded from a config file,
    # an external service, or a more sophisticated NLP library.
    _PROFANE_WORDS = {
        "ass", "bitch", "cunt", "fuck", "shit", "damn", "hell", "piss", "bastard"
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        If the input `data` is a string, it replaces known profane words
        (case-insensitively) with asterisks. The length of the asterisk
        replacement matches the length of the original profane word.
        If `data` is not a string, it logs a warning and returns the data unchanged.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing pipeline.
                                       Currently not used by this node but available.

        Returns:
            Any: The processed data (string with profanity filtered, or original data
                 if not a string).
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data "
                f"(type: {type(data).__name__}). Skipping profanity filtering."
            )
            return data

        processed_text = data
        for word in self._PROFANE_WORDS:
            # Create a regex pattern for the word, case-insensitive, whole word match.
            # Using re.escape to handle special characters if present in words.
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            
            # Replace occurrences with asterisks matching the length of the word
            processed_text = pattern.sub(lambda m: '*' * len(m.group(0)), processed_text)
            
        logger.debug(f"[{self.node_name}] Successfully filtered profanity from data.")
        return processed_text
