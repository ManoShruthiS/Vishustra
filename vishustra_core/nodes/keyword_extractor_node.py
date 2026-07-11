import logging
import re
from typing import Any, Dict, List, Set
from collections import Counter

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract relevant keywords from an input text.

    This node performs basic text cleaning, tokenization, stop-word removal,
    and frequency analysis to identify and return the most significant keywords.
    It is configurable via the `context` dictionary for parameters like stop words,
    minimum word length, and the number of keywords to return.
    """

    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "in", "on", "at", "for", "with", "as",
        "by", "from", "up", "out", "down", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
        "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
        "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
        "won", "wouldn", "about", "above", "after", "again", "against", "among",
        "around", "before", "below", "between", "both", "but", "could", "did",
        "do", "does", "doing", "during", "each", "few", "had", "has", "have",
        "having", "he", "her", "here", "hers", "herself", "him", "himself",
        "his", "how", "if", "into", "its", "itself", "me", "more", "must",
        "my", "myself", "nor", "off", "our", "ours", "ourselves", "out", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this",
        "those", "through", "until", "very", "we", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "you", "your",
        "yours", "yourself", "yourselves"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "keyword_extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        The `data` input is expected to be a string containing the text from
        which keywords need to be extracted. The `context` dictionary can
        be used to customize the extraction parameters.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary for configuration.
                Recognized keys:
                - 'stop_words' (Set[str], optional): A custom set of stop words
                  to filter out. Defaults to an internal predefined list if not provided.
                - 'min_word_length' (int, optional): The minimum length a word
                  must have to be considered a potential keyword. Defaults to 3.
                - 'num_keywords' (int, optional): The maximum number of top
                  keywords to return. Defaults to 5.

        Returns:
            List[str]: A list of unique keywords, sorted by frequency in descending order.

        Raises:
            ValueError: If the `data` input is not a string.
            TypeError: If configuration values in `context` are of incorrect types.
        """
        logger.info(f"[{self.node_name}] Initiating keyword extraction.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'.")
            raise ValueError(f"KeywordExtractorNode requires 'data' to be a string, but received {type(data).__name__}.")

        text = data.lower()

        # --- Retrieve and validate configuration from context ---
        stop_words = context.get('stop_words', self._DEFAULT_STOP_WORDS)
        if not isinstance(stop_words, set):
            logger.error(f"[{self.node_name}] Invalid type for 'stop_words' in context. Expected 'set', got '{type(stop_words).__name__}'.")
            raise TypeError("Context parameter 'stop_words' must be a set of strings.")

        min_word_length = context.get('min_word_length', 3)
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.error(f"[{self.node_name}] Invalid value or type for 'min_word_length' in context. Expected a positive integer, got '{min_word_length}'.")
            raise TypeError("Context parameter 'min_word_length' must be a positive integer.")

        num_keywords = context.get('num_keywords', 5)
        if not isinstance(num_keywords, int) or num_keywords < 0:
            logger.error(f"[{self.node_name}] Invalid value or type for 'num_keywords' in context. Expected a non-negative integer, got '{num_keywords}'.")
            raise TypeError("Context parameter 'num_keywords' must be a non-negative integer.")

        # --- Text processing ---
        # Tokenize and filter words based on minimum length and alphabetical characters only
        # The regex ensures words are at least 'min_word_length' characters long and composed of a-z.
        words = re.findall(r'\b[a-z]{%d,}\b' % min_word_length, text)

        # Filter out stop words
        filtered_words = [word for word in words if word not in stop_words]

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Get the top N most common words as keywords
        keywords = [word for word, _ in word_counts.most_common(num_keywords)]

        logger.info(f"[{self.node_name}] Successfully extracted {len(keywords)} keywords.")
        return keywords
