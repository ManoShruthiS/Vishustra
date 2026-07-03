import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A processing node that extracts keywords from a given text string.

    This node performs a basic simulation of keyword extraction by:
    1. Lowercasing the input text.
    2. Tokenizing words and filtering out non-alphanumeric characters.
    3. Filtering out common stop words (configurable via context).
    4. Filtering out words shorter than a minimum length (configurable via context).
    5. Returning a sorted list of unique remaining words.

    Configuration parameters that can be passed in the `context` dictionary:
    - `stop_words` (list or set of str): A collection of words to ignore.
      If not provided or invalid, a default set of common English stop words is used.
    - `min_word_length` (int): The minimum length a word must have to be considered a keyword.
      If not provided or invalid, defaults to 3.
    """

    def __init__(self):
        """
        Initializes the KeywordExtractor node.
        No specific state is managed at the instance level for this basic implementation.
        """
        super().__init__()
        logger.debug(f"[{self.node_name}] Initializing KeywordExtractor node.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       and configuration parameters for the node.

        Returns:
            List[str]: A sorted list of unique extracted keywords.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only string, returning empty keyword list.")
            return []

        text = data.lower()

        # --- Configure Stop Words ---
        stop_words_config = context.get("stop_words")
        configured_stop_words: Set[str]
        if isinstance(stop_words_config, (list, set)):
            configured_stop_words = set(stop_words_config)
            logger.debug(f"[{self.node_name}] Using {len(configured_stop_words)} stop words from context.")
        else:
            if stop_words_config is not None:
                logger.warning(
                    f"[{self.node_name}] 'stop_words' in context has invalid type '{type(stop_words_config).__name__}'. Expected 'list' or 'set'. Using default internal stop words."
                )
            # Default common English stop words
            configured_stop_words = {
                "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for", "of", "in", "on", "with", "to",
                "from", "at", "it", "this", "that", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
                "them", "my", "your", "his", "its", "our", "their", "be", "has", "have", "had", "do", "does", "did",
                "not", "can", "will", "would", "should", "could", "get", "go", "just", "like", "know", "see", "think",
                "time", "up", "down", "out", "about", "all", "any", "some", "most", "many", "other", "much", "more",
                "no", "yes", "than", "then", "very", "also", "well", "only", "such", "said", "say", "get", "make", "made"
            }
            logger.debug(f"[{self.node_name}] Using default internal stop words ({len(configured_stop_words)} words).")

        # --- Configure Minimum Word Length ---
        min_word_length: int = context.get("min_word_length", 3)
        if not isinstance(min_word_length, int) or min_word_length < 1:
            if context.get("min_word_length") is not None:
                logger.warning(
                    f"[{self.node_name}] 'min_word_length' in context is invalid ({min_word_length}, type: {type(min_word_length).__name__}). Using default of 3."
                )
            min_word_length = 3
        else:
            logger.debug(f"[{self.node_name}] Using min_word_length '{min_word_length}' from context.")

        # --- Basic Tokenization and Filtering ---
        # Replace non-alphanumeric characters with spaces to facilitate word splitting
        processed_text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = [word for word in processed_text.split() if word.strip()]

        extracted_keywords_list: List[str] = []
        for word in words:
            # Filter out stop words and short words
            if word not in configured_stop_words and len(word) >= min_word_length:
                extracted_keywords_list.append(word)
        
        # Return unique keywords, sorted for consistent output
        unique_keywords = sorted(list(set(extracted_keywords_list)))
        
        logger.info(
            f"[{self.node_name}] Successfully processed text and extracted {len(unique_keywords)} keywords."
        )
        return unique_keywords