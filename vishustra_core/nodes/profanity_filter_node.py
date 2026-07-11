import logging
import re
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project root or sys.path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanities from text data.

    This node identifies and replaces predefined profane words within a given
    text string with asterisks, maintaining the original length of the word.
    It performs case-insensitive matching and handles whole word boundaries
    to prevent unintended replacements within larger words.
    """

    # For a production system, this list would typically be configurable,
    # loaded from a file, or managed by a more sophisticated profanity library.
    # Using a set for efficient lookups.
    _PROFANE_WORDS = {
        "ass", "bastard", "bitch", "cunt", "damn", "fuck", "hell", "piss", "shit",
        "tits", "wank"
    }
    _REPLACEMENT_CHAR = "*"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        If the input `data` is a string, it replaces specified profane words
        with asterisks. The replacement maintains the original length of the
        profane word (e.g., "fuck" becomes "****"). Matches are case-insensitive
        and respect word boundaries.

        If `data` is not a string, it logs a warning and returns the data unchanged.

        Args:
            data: The input data, expected to be a string to be filtered.
            context: A dictionary containing contextual information for the pipeline
                     (not directly used by this node but part of the interface).

        Returns:
            The processed data (string with profanity filtered) or the
            original data if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data).__name__}. "
                "Profanity filtering skipped. Returning original data."
            )
            return data

        processed_text = data
        for word in self._PROFANE_WORDS:
            # Create a regex pattern for whole words, case-insensitive.
            # \b ensures word boundaries, re.escape handles special regex characters in the word.
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Replace the word with asterisks of the same length as the matched word.
            # The replacement function ensures each matched word gets its length-appropriate asterisks.
            processed_text = re.sub(
                pattern,
                lambda m: self._REPLACEMENT_CHAR * len(m.group(0)), # m.group(0) is the actual matched word
                processed_text,
                flags=re.IGNORECASE
            )
        
        logger.info(f"[{self.node_name}] Successfully processed data for profanity filtering.")
        return processed_text