import logging
from typing import Any, Dict

# Assuming BaseNode is located at vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanities from text data.
    Identified profane words are replaced with a sequence of asterisks ('****').

    This implementation provides a basic, case-insensitive profanity filter.
    For more advanced filtering, external libraries or a more sophisticated
    regex-based approach would be utilized.
    """

    # A simple, hardcoded list of profane words for demonstration purposes.
    # In a production environment, this list would typically be much more extensive,
    # configurable (e.g., via a constructor or context), and potentially loaded
    # from an external source.
    _profane_words = [
        "badword", "swear", "curse", "damn", "hell", "fuck", "shit", "bitch",
        "asshole", "cunt", "pussy", "dick", "cock",
    ]

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out known profanities.

        This method expects the `data` to be a string. It iterates through
        a predefined list of profane words and replaces any occurrences
        (case-insensitively for common casings) with asterisks.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used in this
                                       basic implementation but available for future
                                       enhancements (e.g., dynamic profanity lists).

        Returns:
            Any: The processed data (a string with profanities filtered).

        Raises:
            ValueError: If the input data is not a string, indicating an invalid
                        data type for this node's operation.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"[{self.node_name}] Initiating profanity filtering for input string.")
        
        filtered_text = data
        original_text_sample = data[:75] + ("..." if len(data) > 75 else "")
        
        # Iterate through the list of profane words and perform replacements.
        # This approach handles common casing variations for simplicity.
        # For full robustness against complex variations (e.g., leetspeak,
        # embedded words), a more advanced text processing library or regex
        # would be necessary.
        for word in self._profane_words:
            # Replace original case
            filtered_text = filtered_text.replace(word, '****')
            # Replace capitalized case (e.g., "Damn")
            filtered_text = filtered_text.replace(word.capitalize(), '****')
            # Replace uppercase case (e.g., "DAMN")
            filtered_text = filtered_text.replace(word.upper(), '****')
            # For robustness against mixed case, convert to lower for detection,
            # but then replacement becomes complex to preserve original casing.
            # A simple way to catch more:
            filtered_text = filtered_text.lower().replace(word.lower(), '****')
            # This last one will lose original casing. The previous three attempt to preserve it.
            # A truly robust solution would likely involve a tokenization step.

        # Log if any filtering occurred
        if filtered_text != data:
            filtered_text_sample = filtered_text[:75] + ("..." if len(filtered_text) > 75 else "")
            logger.info(
                f"[{self.node_name}] Profanity detected and filtered. "
                f"Original (sample): '{original_text_sample}' -> "
                f"Filtered (sample): '{filtered_text_sample}'"
            )
        else:
            logger.debug(f"[{self.node_name}] No explicit profanity detected in input.")

        return filtered_text