import logging
import re
from typing import Any, Dict, List, Set

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text.

    This node tokenizes the input text, converts words to lowercase, filters
    out common stop words and words shorter than a specified minimum length,
    and returns a unique list of identified keywords.
    """

    def __init__(self, min_word_length: int = 4, stop_words: Set[str] = None):
        """
        Initializes the KeywordExtractorNode.

        Args:
            min_word_length (int): The minimum length a word must have to be considered a keyword.
            stop_words (Set[str], optional): A set of words to be excluded from keywords.
                                              Defaults to a common English stop word set if None.
        """
        if not isinstance(min_word_length, int) or min_word_length < 1:
            raise ValueError("min_word_length must be a positive integer.")
        if stop_words is not None and not isinstance(stop_words, set):
            raise TypeError("stop_words must be a set of strings or None.")

        self._min_word_length = min_word_length
        self._stop_words = stop_words if stop_words is not None else self._default_stop_words()
        logger.debug(
            f"KeywordExtractorNode initialized with min_word_length={self._min_word_length} "
            f"and {len(self._stop_words)} stop words."
        )

    def _default_stop_words(self) -> Set[str]:
        """Provides a default set of common English stop words."""
        # This set can be expanded or made configurable via external data in a real application
        return {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "once", "here", "there", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "s", "t", "can", "will", "just", "don", "should", "now",
            "is", "am", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "having", "do", "does", "did", "doing", "would",
            "could", "shall", "may", "might", "must", "it", "its", "me", "my",
            "myself", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "we", "us", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "this", "that", "these", "those",
            "i", "what", "which", "who", "whom", "of", "also", "into", "through",
            "during", "before", "after", "above", "below", "between", "among",
            "while", "as", "much", "many", "even", "much", "very", "can", "will",
            "said", "say", "says", "like"
        }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data.

        The input `data` is expected to be a string. It is tokenized,
        converted to lowercase, and filtered based on minimum word length
        and a configurable set of stop words.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing (not used by this node, but required
                                       by the BaseNode interface).

        Returns:
            List[str]: A sorted list of unique keywords extracted from the text,
                       converted to lowercase. An empty list is returned if no
                       keywords are found or if the input text is empty.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input type for KeywordExtractorNode. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"KeywordExtractorNode requires string input, but received {type(data).__name__}.")

        text = data.strip()
        if not text:
            logger.warning("KeywordExtractorNode received an empty string for processing, returning empty list.")
            return []

        logger.info(f"Starting keyword extraction for input of length {len(text)}.")

        # Tokenize words using a regex that handles alphanumeric characters,
        # and allows internal apostrophes or hyphens (e.g., "AI-driven", "don't").
        # Converts to lowercase directly.
        words = re.findall(r'\b[a-z0-9](?:[a-z0-9\'-]*[a-z0-9])?\b', text.lower())

        keywords_set = set()
        for word in words:
            if word not in self._stop_words and len(word) >= self._min_word_length:
                keywords_set.add(word)

        result = sorted(list(keywords_set))
        logger.info(f"Finished keyword extraction. Found {len(result)} unique keywords.")
        return result
