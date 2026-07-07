
import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text.

    This node tokenizes the input text, converts it to lowercase, removes
    stopwords, and filters words based on their length. It returns a sorted list
    of unique keywords.

    Configuration via context:
    - `stop_words` (Set[str] or List[str]): A collection of words to ignore.
      Defaults to a comprehensive English stopword list if not provided.
    - `min_keyword_length` (int): The minimum character length for a word to be
      considered a keyword. Defaults to 3.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing runtime context
                                      and configuration parameters for the node.

        Returns:
            List[str]: A sorted list of unique extracted keywords.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'min_keyword_length' in context is invalid.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for KeywordExtractorNode. "
                f"Expected 'str', got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.lower()

        # Define default stop words. This list can be extended or replaced via context.
        default_stop_words: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for",
            "with", "to", "of", "in", "on", "at", "by", "from", "as", "it", "its",
            "that", "this", "he", "she", "we", "you", "they", "them", "us", "him",
            "her", "i", "me", "my", "your", "our", "their", "will", "would", "can",
            "could", "should", "may", "might", "have", "has", "had", "do", "does",
            "did", "not", "no", "yes", "be", "been", "being", "such", "also", "many",
            "much", "more", "most", "some", "any", "all", "each", "every", "other",
            "another", "when", "where", "why", "how", "what", "who", "whom", "which",
            "if", "then", "than", "so", "up", "down", "out", "about", "into", "through",
            "during", "before", "after", "above", "below", "between", "among", "within",
            "without", "over", "under", "again", "further", "once", "here", "there",
            "just", "don", "shouldn", "now", "s", "t", "m", "o", "re", "ve", "y", "d", "ll",
            "am", "is", "are", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "would", "should", "could"
        }

        # Retrieve stop words from context, converting to a set for efficient lookup
        stop_words_config = context.get("stop_words", default_stop_words)
        if isinstance(stop_words_config, (list, tuple)):
            stop_words = set(stop_words_config)
        elif isinstance(stop_words_config, set):
            stop_words = stop_words_config
        else:
            logger.warning(
                f"Invalid type for 'stop_words' in context ({type(stop_words_config).__name__}). "
                "Expected a list, tuple, or set of strings. Using default stop words."
            )
            stop_words = default_stop_words

        # Retrieve minimum keyword length from context
        min_keyword_length = context.get("min_keyword_length", 3)
        if not isinstance(min_keyword_length, int) or min_keyword_length <= 0:
            error_msg = (
                f"Invalid 'min_keyword_length' in context: '{min_keyword_length}'. "
                "Expected a positive integer."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Use regex to find words, handling various punctuation and ensuring word boundaries
        words = re.findall(r'\b\w+\b', text)

        extracted_keywords = set()
        for word in words:
            if word not in stop_words and len(word) >= min_keyword_length:
                extracted_keywords.add(word)

        result = sorted(list(extracted_keywords))
        logger.info(f"KeywordExtractorNode processed data and extracted {len(result)} unique keywords.")
        return result

