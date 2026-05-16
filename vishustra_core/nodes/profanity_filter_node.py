import logging
import re
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to identify and mask profane or sensitive language
    within text data using configurable patterns.
    """

    def __init__(
        self, 
        blocked_words: Optional[List[str]] = None, 
        replacement_token: str = "****",
        case_sensitive: bool = False
    ):
        """
        Initializes the ProfanityFilterNode.

        :param blocked_words: A list of strings to be filtered. Defaults to a placeholder list.
        :param replacement_token: The string used to mask detected profanity.
        :param case_sensitive: Boolean indicating if the filter should respect casing.
        """
        self._blocked_words = blocked_words or ["offensive_term_a", "offensive_term_b"]
        self._replacement_token = replacement_token
        
        flags = 0 if case_sensitive else re.IGNORECASE
        if self._blocked_words:
            # Construct a regex pattern for efficient matching
            pattern = r'\b(' + '|'.join(map(re.escape, self._blocked_words)) + r')\b'
            self._regex = re.compile(pattern, flags)
        else:
            self._regex = None

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Sanitizes the input data. Expects a string or a dictionary containing text fields.

        :param data: The input data to process (expected to be a string or contain strings).
        :param context: Execution context for the node.
        :return: The sanitized data.
        """
        try:
            if data is None:
                logger.debug("Received null data; skipping filtration.")
                return None

            if not self._regex:
                logger.warning("No blocked words configured; passing data through unchanged.")
                return data

            if isinstance(data, str):
                return self._sanitize_text(data)
            
            if isinstance(data, dict):
                return {k: (self._sanitize_text(v) if isinstance(v, str) else v) for k, v in data.items()}

            logger.info(f"Input type {type(data).__name__} is not directly supported for masking. Returning as is.")
            return data

        except Exception as e:
            logger.error(f"Failed to process data in {self.node_name}: {e}", exc_info=True)
            raise

    def _sanitize_text(self, text: str) -> str:
        """Helper method to apply regex substitution."""
        return self._regex.sub(self._replacement_token, text)

    def __repr__(self) -> str:
        return f"<{self.node_name}(words_count={len(self._blocked_words)})>"

