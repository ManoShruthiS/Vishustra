import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure
# from vishustra_core.nodes.base_node import BaseNode
# For standalone execution/testing, we'll use the definition provided in the prompt
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
    A Vishustra node that filters profanity from text data.

    This node identifies and replaces predefined profanity words within input
    strings or lists of strings, replacing them with a masked equivalent (e.g., ****).
    It handles case-insensitivity and aims for whole-word matching to avoid
    filtering parts of innocent words.
    """

    # A simple, illustrative list of profanity for demonstration.
    # In a real-world scenario, this would be much more extensive,
    # potentially loaded from a configuration file or a dedicated service.
    _PROFANITY_LIST: List[str] = [
        "fuck", "shit", "bitch", "asshole", "damn", "cunt", "motherfucker",
        "pussy", "dick", "cock", "bastard"
    ]

    def __init__(self, replacement_char: str = '*') -> None:
        """
        Initializes the ProfanityFilterNode.

        Args:
            replacement_char: The character to use for masking profanity.
                              Defaults to '*'.
        """
        self._replacement_char = replacement_char
        # Pre-compile regex patterns for efficiency and robust whole-word matching
        self._profanity_patterns: List[re.Pattern] = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self._PROFANITY_LIST
        ]
        logger.debug(f"ProfanityFilterNode initialized with replacement char: '{replacement_char}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def _filter_text(self, text: str) -> str:
        """
        Applies the profanity filter to a single string.

        Args:
            text: The input string to filter.

        Returns:
            The filtered string with profanity masked.
        """
        filtered_text = text
        for pattern in self._profanity_patterns:
            # Replace all occurrences of the profanity pattern
            filtered_text = pattern.sub(lambda m: self._replacement_char * len(m.group(0)), filtered_text)
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     (Not directly used by this node but part of the BaseNode API).

        Returns:
            The processed data with profanity filtered.
            Returns the original data if it's not a string or list of strings,
            after logging a warning.

        Raises:
            Exception: Catches and logs any unexpected errors during processing
                       and re-raises them for upstream handling.
        """
        logger.info(f"[{self.node_name}] Starting profanity filtering for data type: {type(data)}")
        try:
            if isinstance(data, str):
                processed_data = self._filter_text(data)
                if processed_data != data:
                    logger.debug(f"[{self.node_name}] Profanity detected and filtered in string data.")
                return processed_data
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                processed_data = [self._filter_text(item) for item in data]
                if any(p != o for p, o in zip(processed_data, data)):
                    logger.debug(f"[{self.node_name}] Profanity detected and filtered in list of strings.")
                return processed_data
            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported data type for profanity filtering: {type(data)}. "
                    "Expected str or List[str]. Returning original data."
                )
                return data
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during processing: {e}")
            # Depending on framework policy, you might re-raise, return an error object, or original data
            raise # Re-raise to ensure upstream orchestration handles the failure