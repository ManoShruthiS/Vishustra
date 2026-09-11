import logging
import re
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists and contains BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out common profane words from text data.
    It replaces identified profanities with a sequence of asterisks ('***') to sanitize content.
    """

    # A curated list of profane words. In a production system, this list would
    # typically be dynamically loaded from a configuration, external service,
    # or a more sophisticated lexicon for maintainability and customization.
    _PROFANE_WORDS = [
        "fuck", "shit", "bitch", "asshole", "cunt", "motherfucker",
        "damn", "piss", "cock", "dick", "faggot", "whore", "bastard"
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, identifying and replacing profane words.

        This method expects `data` to be a string containing the text to be filtered.
        If `data` is not a string, a warning is logged, and the original data
        is returned without modification to avoid interrupting the pipeline.

        Args:
            data: The input data, ideally a string, which will be scanned for profanities.
            context: A dictionary containing contextual information relevant to the
                     current processing flow, which can be utilized by the node.

        Returns:
            The processed data with all identified profane words replaced by '***'.
            If the input `data` was not a string, the original `data` is returned.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data for profanity filtering. "
                f"Expected str, got {type(data).__name__}. Returning data unchanged."
            )
            return data

        processed_text = data

        for word in self._PROFANE_WORDS:
            # Construct a regex pattern for case-insensitive, whole-word matching.
            # `re.escape` handles any special regex characters within `word`.
            # `\b` ensures we match whole words only (e.g., "fuck" but not "fuchsia").
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Perform the replacement globally and case-insensitively.
            processed_text = re.sub(pattern, '***', processed_text, flags=re.IGNORECASE)

        logger.debug(f"[{self.node_name}] Successfully processed and sanitized text data.")
        return processed_text