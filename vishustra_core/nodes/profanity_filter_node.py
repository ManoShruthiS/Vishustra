import logging
import re
from typing import Any, Dict, List, Optional

# Assuming this import path based on project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node designed to filter profanity from input text.

    This node identifies and replaces predefined profane words within an input string
    with a specified masking string (e.g., '***'). It supports configurable profanity lists
    and replacement strings, and performs case-insensitive, whole-word matching.
    """

    def __init__(self, profanity_list: Optional[List[str]] = None, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode with a list of profane words and a replacement string.

        Args:
            profanity_list (Optional[List[str]]): An optional list of words to be considered profane.
                                                  If None, a default list of common profanities will be used.
            replacement_string (str): The string to substitute for each detected profane word.
                                      Defaults to '***'.
        """
        # Ensure profanity list words are lowercased for consistent matching later
        self._profanity_list = [word.lower() for word in profanity_list] if profanity_list else self._get_default_profanity_list()
        self._replacement_string = replacement_string
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._profanity_list)} "
            f"profanity words and replacement string: '{self._replacement_string}'"
        )

    def _get_default_profanity_list(self) -> List[str]:
        """
        Provides a default, hardcoded list of profanity words.

        In a production environment, this list would typically be loaded from a
        secure configuration service, a dedicated dataset, or a dynamic source.

        Returns:
            List[str]: A list of default profanity words.
        """
        return [
            "ass", "asshole", "bitch", "bastard", "crap", "damn", "dick", "fuck",
            "hell", "motherfucker", "shit", "wanker", "prick", "cunt", "pussy"
        ]

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        If the input `data` is a string, it iterates through the configured
        profanity list and replaces any matching whole words (case-insensitively)
        with the specified `replacement_string`. If `data` is not a string,
        a warning is logged, and the original data is returned unchanged.

        Args:
            data (Any): The input data. Expected to be a string for filtering.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       This node does not currently utilize the context.

        Returns:
            Any: The sanitized string if `data` was a string, otherwise the
                 original `data` unchanged.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering skipped. Returning original data."
            )
            return data

        processed_text = data
        for word in self._profanity_list:
            # Use regex for robust, case-insensitive, whole-word matching.
            # \b ensures word boundaries, re.escape handles special characters in profanity words.
            pattern = r'\b' + re.escape(word) + r'\b'
            processed_text = re.sub(pattern, self._replacement_string, processed_text, flags=re.IGNORECASE)

        logger.info(f"[{self.node_name}] Successfully applied profanity filter to text.")
        return processed_text
