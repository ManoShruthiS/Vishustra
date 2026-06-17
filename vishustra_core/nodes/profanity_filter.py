import logging
import re
from typing import Any, Dict

# Assuming the base_node is located at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node responsible for filtering profanities from text data.
    It identifies and replaces common profanities within the input text with a
    masked string (e.g., '****'), ensuring content compliance.
    """

    # A simple, static list of profanities. In a production environment, this list
    # would typically be loaded from external configuration, a database, or managed
    # by a dedicated content moderation service. Using a frozenset for efficient,
    # immutable lookup and better performance.
    _PROFANITIES: frozenset[str] = frozenset([
        "ass", "bitch", "bastard", "cock", "cunt", "damn", "dick", "fuck",
        "motherfucker", "pussy", "shit", "slag", "whore"
    ])
    
    _MASK_STRING = "****" # The string used to replace detected profanities

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to detect and filter out profanities.
        
        This method expects the input `data` to be a string. If `data` is not
        a string, a warning is logged, and the original `data` is returned
        unmodified. Detected profanities are replaced with `_MASK_STRING`.

        Args:
            data: The input data to be processed, typically a string containing text.
            context: A dictionary holding additional contextual information relevant
                     to the current processing pipeline run. (Not directly used by
                     this specific node's logic but part of the BaseNode contract).

        Returns:
            The processed data with profanities filtered, or the original data
            if it was not a string or no profanities were found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering skipped. Returning original data."
            )
            return data

        processed_text = data
        total_detected_count = 0
        
        # Iterate through the list of known profanities
        for profanity in self._PROFANITIES:
            # Construct a regex pattern for the profanity with word boundaries.
            # re.escape() handles any special regex characters within the profanity itself.
            # \b ensures that only whole words are matched (e.g., 'ass' won't match in 'associate').
            pattern = r'\b' + re.escape(profanity) + r'\b'
            
            # Use re.subn to replace all occurrences of the profanity, case-insensitively.
            # re.subn returns a tuple: (new_string, number_of_replacements).
            new_text, num_replacements = re.subn(pattern, self._MASK_STRING, processed_text, flags=re.IGNORECASE)
            
            if num_replacements > 0:
                total_detected_count += num_replacements
                # Log individual detections at debug level for detailed tracing
                logger.debug(
                    f"[{self.node_name}] Masked '{profanity}' {num_replacements} time(s)."
                )
                processed_text = new_text # Update the text for subsequent replacements

        if total_detected_count > 0:
            logger.info(
                f"[{self.node_name}] Successfully filtered {total_detected_count} profanity instances. "
                f"Original data length: {len(data)}, Filtered data length: {len(processed_text)}."
            )
        else:
            logger.debug(f"[{self.node_name}] No profanities detected in the input data.")

        return processed_text
