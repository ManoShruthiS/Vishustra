import logging
import re
from typing import Any, Dict, List, Set, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to identify and mask profanity within textual data.
    
    This node scans input strings or lists of strings against a configurable list 
    of prohibited terms and replaces them with a masking character.
    """

    def __init__(self, default_banned_words: List[str] = None):
        """
        Initializes the node with an optional default list of banned words.
        """
        self._default_banned_words = set(default_banned_words) if default_banned_words else set()

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to sanitize profane content.
        
        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): Metadata and configuration, including:
                - 'extra_banned_words': List[str] (Optional)
                - 'mask_char': str (Optional, defaults to '*')
                - 'case_sensitive': bool (Optional, defaults to False)

        Returns:
            Any: The sanitized data in its original structure.
        
        Raises:
            TypeError: If the input data is not a string or list of strings.
        """
        logger.debug(f"Executing {self.node_name} processing logic.")

        try:
            if data is None:
                logger.warning("Received null data in ProfanityFilterNode.")
                return None

            # Extract configuration from context
            mask_char = context.get("mask_char", "*")
            case_sensitive = context.get("case_sensitive", False)
            extra_words = set(context.get("extra_banned_words", []))
            
            # Combine word sets
            banned_words = self._default_banned_words.union(extra_words)
            
            if not banned_words:
                logger.debug("No banned words defined. Skipping filtering.")
                return data

            if isinstance(data, str):
                return self._filter_text(data, banned_words, mask_char, case_sensitive)
            elif isinstance(data, list):
                return [
                    self._filter_text(item, banned_words, mask_char, case_sensitive) 
                    if isinstance(item, str) else item 
                    for item in data
                ]
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                raise TypeError(f"ProfanityFilterNode expects str or list, got {type(data)}")

        except Exception as e:
            logger.exception(f"Error encountered during profanity filtering: {str(e)}")
            raise

    def _filter_text(self, text: str, words: Set[str], mask: str, case_sensitive: bool) -> str:
        """
        Internal utility to mask words in a single string.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # Sort words by length descending to prevent partial matches of longer phrases
        sorted_words = sorted(list(words), key=len, reverse=True)
        
        # Build a regex pattern for all banned words
        # Using word boundaries (\b) to ensure we don't match substrings inside safe words
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_words)) + r')\b', flags=flags)

        def replace_match(match):
            word = match.group(0)
            return mask * len(word)

        return pattern.sub(replace_match, text)

    def __repr__(self) -> str:
        return f"<{self.node_name}(words_count={len(self._default_banned_words)})>"