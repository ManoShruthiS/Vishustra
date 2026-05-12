import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A moderation node designed to identify and mask profane language within 
    textual data before passing it to subsequent LLM chain nodes.
    """

    DEFAULT_BANNED_WORDS = [
        "badword1", "badword2", "offensive_term" 
    ]  # In a production environment, this would be loaded from an external config or encrypted source.

    def __init__(self, custom_words: Optional[List[str]] = None, replacement: str = "****"):
        """
        Initializes the ProfanityFilterNode.

        :param custom_words: An optional list of additional words to filter.
        :param replacement: The string used to mask identified profanity.
        """
        self._banned_words = set(self.DEFAULT_BANNED_WORDS)
        if custom_words:
            self._banned_words.update(custom_words)
        
        self._replacement = replacement
        # Pre-compile regex for performance
        pattern_str = r'\b(' + '|'.join(map(re.escape, self._banned_words)) + r')\b'
        self._pattern = re.compile(pattern_str, re.IGNORECASE)

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "Profanity Filter Node"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, replacing banned words with a replacement mask.

        :param data: The input data, expected to be a string or a dictionary containing text.
        :param context: Execution context, can be used to override replacement settings dynamically.
        :return: The sanitized data.
        """
        try:
            replacement = context.get("profanity_replacement", self._replacement)
            
            if isinstance(data, str):
                return self._sanitize_text(data, replacement)
            
            if isinstance(data, dict):
                return {k: (self._sanitize_text(v, replacement) if isinstance(v, str) else v) 
                        for k, v in data.items()}

            if isinstance(data, list):
                return [self._sanitize_text(item, replacement) if isinstance(item, str) else item 
                        for item in data]

            logger.warning(f"[{self.node_name}] Received unsupported data type: {type(data)}. Skipping transformation.")
            return data

        except Exception as e:
            logger.error(f"[{self.node_name}] Error during processing: {str(e)}", exc_info=True)
            raise RuntimeError(f"ProfanityFilterNode failed to process data: {e}")

    def _sanitize_text(self, text: str, replacement: str) -> str:
        """
        Internal helper to execute the regex substitution.
        """
        if not text:
            return text
        return self._pattern.sub(replacement, text)

# End of file