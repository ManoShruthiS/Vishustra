import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node designed to filter out specified profane words from input strings.
    It replaces identified profane words with '***' to sanitize content.
    """

    # For this initial simulation, a hardcoded list of profanities is used.
    # In a production environment, this list would typically be configurable,
    # loaded from a persistent store, or managed via node configuration.
    _profane_words = [
        "shit", "fuck", "damn", "asshole", "bitch", "cunt", "bastard",
        "pussy", "dick", "motherfucker", "cock", "tits", "wanker"
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, replacing any detected profane words with '***'.

        Args:
            data: The input data, which is expected to be a string.
            context: A dictionary containing contextual information for processing.
                     This node does not currently utilize the context, but it's
                     available for future enhancements (e.g., dynamic word lists).

        Returns:
            The processed string with profane words filtered out.

        Raises:
            TypeError: If the input 'data' is not a string, enforcing strict input types
                       for predictable pipeline behavior.
        """
        if not isinstance(data, str):
            logger.error(
                f"ProfanityFilterNode received unsupported data type: {type(data).__name__}. "
                "This node exclusively processes string input."
            )
            raise TypeError(
                f"ProfanityFilterNode requires string input, but received type {type(data).__name__}."
            )

        original_data = data
        processed_data = data

        for word in self._profane_words:
            # Construct a regex pattern for whole-word matching.
            # re.escape() is used to handle special characters within profane words safely.
            # \b ensures word boundaries, preventing partial matches (e.g., 'ass' in 'associate').
            pattern = r'\b' + re.escape(word) + r'\b'
            # Perform a case-insensitive substitution.
            processed_data = re.sub(pattern, '***', processed_data, flags=re.IGNORECASE)

        if original_data != processed_data:
            # Log significant changes, truncating long strings for log readability.
            log_original_preview = original_data[:100] + ('...' if len(original_data) > 100 else '')
            log_filtered_preview = processed_data[:100] + ('...' if len(processed_data) > 100 else '')
            logger.info(
                f"Profanity filter applied. "
                f"Original (preview): '{log_original_preview}' | "
                f"Filtered (preview): '{log_filtered_preview}'"
            )
        else:
            logger.debug("No profane words detected or filtered in the input data.")

        return processed_data
