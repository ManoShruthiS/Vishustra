import logging
import re
from typing import Any, Dict, List, Optional, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to sanitize string data by identifying and 
    masking profanity based on a configurable blocklist.
    """

    DEFAULT_BLOCKLIST = {
        "badword1", "badword2", "inappropriate", "toxic"
    }

    def __init__(self, custom_blocklist: Optional[List[str]] = None, mask_char: str = "*"):
        """
        Initializes the ProfanityFilterNode.

        Args:
            custom_blocklist: Optional list of strings to filter.
            mask_char: The character used to mask detected profanity.
        """
        self._blocklist: Set[str] = set(custom_blocklist) if custom_blocklist else self.DEFAULT_BLOCKLIST
        self._mask_char = mask_char
        # Pre-compile regex for performance
        self._pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, self._blocklist)) + r')\b', 
            flags=re.IGNORECASE
        )

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node type."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, applying masking to any detected profanity.

        Args:
            data: The input to process. Expected to be a string or a dictionary 
                  containing text fields.
            context: Execution context containing metadata or runtime overrides.

        Returns:
            The sanitized version of the input data.

        Raises:
            TypeError: If the data format is unsupported.
        """
        try:
            if isinstance(data, str):
                return self._sanitize_text(data)
            
            if isinstance(data, dict):
                return {
                    key: self._sanitize_text(value) if isinstance(value, str) else value 
                    for key, value in data.items()
                }

            logger.warning(f"[{self.node_name}] Received unsupported data type: {type(data)}. Skipping transformation.")
            return data

        except Exception as e:
            logger.error(f"[{self.node_name}] Error during processing: {str(e)}")
            raise

    def _sanitize_text(self, text: str) -> str:
        """
        Performs regex-based replacement of blocked words.
        """
        def replace_match(match: re.Match) -> str:
            word = match.group(0)
            return self._mask_char * len(word)

        sanitized, count = self._pattern.subn(replace_match, text)
        
        if count > 0:
            logger.info(f"[{self.node_name}] Masked {count} instances of profanity.")
            
        return sanitized