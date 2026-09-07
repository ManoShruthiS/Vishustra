
import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out common profanities from text data.
    It replaces identified profanity words with '***' while attempting to preserve
    the original text structure and casing where possible outside the replaced words.
    """

    # A curated set of common profanity words for demonstration purposes.
    # In a production environment, this list would typically be externalized
    # to a configuration service, database, or a dedicated dictionary file
    # for easy updates and language support.
    _PROFANITY_LIST = {
        "asshole", "bitch", "bastard", "cunt", "damn", "dick", "fag", "fuck",
        "hell", "motherfucker", "nigger", "piss", "porn", "shit", "slut",
        "tard", "whore"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, identifying and replacing profanity words with '***'.

        The filtering is case-insensitive and ensures only whole words are replaced
        to avoid partial matches within legitimate words (e.g., 'scunthorpe' vs 'cunt').

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information for the processing.
                     This node does not currently utilize the context, but it's
                     available for future extensions.

        Returns:
            The sanitized string with profanities replaced by '***'.

        Raises:
            TypeError: If the input 'data' is not a string, as this node
                       is specifically designed for text processing.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {repr(data)[:100]}..."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        original_data = data
        sanitized_data = data

        for word in self._PROFANITY_LIST:
            # Construct a regex pattern for case-insensitive, whole-word matching.
            # re.escape() handles special characters in the profanity word itself.
            # \b ensures word boundaries, preventing partial word matches.
            pattern = r'\b' + re.escape(word) + r'\b'
            sanitized_data = re.sub(pattern, '***', sanitized_data, flags=re.IGNORECASE)
        
        # Log the transformation for traceability if changes were made.
        if original_data != sanitized_data:
            logger.info(
                "[%s] Data sanitized. Original (first 75 chars): '%s', Sanitized (first 75 chars): '%s'",
                self.node_name, original_data[:75].replace('\n', '\\n'), sanitized_data[:75].replace('\n', '\\n')
            )
        else:
            logger.debug(
                "[%s] No profanity found in data (first 75 chars): '%s'",
                self.node_name, original_data[:75].replace('\n', '\\n')
            )

        return sanitized_data
