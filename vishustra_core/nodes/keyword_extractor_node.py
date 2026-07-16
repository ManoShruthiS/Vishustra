import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node that extracts keywords from a given text.

    This node performs a basic frequency-based keyword extraction,
    filtering out common stop words and words shorter than a specified length.
    It can be configured via the context dictionary to control the number
    of keywords, minimum word length, and a custom set of stop words.
    """

    # A simple set of default English stop words for demonstration purposes.
    # In a production scenario, this list would be more comprehensive,
    # potentially configurable through external means or integrated with
    # a dedicated natural language processing library like NLTK or SpaCy.
    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "not", "no", "yes", "for", "with", "at",
        "by", "from", "up", "down", "on", "off", "in", "out", "over", "under",
        "again", "further", "then", "once", "this", "that", "these", "those",
        "am", "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
        "his", "she", "her", "it", "its", "they", "them", "their", "what",
        "which", "who", "whom", "whose", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such",
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
        "will", "just", "don", "should", "now", "would", "could", "shall"
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data based on frequency.

        The process involves lowercasing the text, tokenizing into words,
        filtering out stop words and short words, and then identifying the
        most frequent remaining words as keywords.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing the text from which to extract keywords.
            context (Dict[str, Any]): A dictionary containing runtime parameters.
                                      Supported parameters include:
                                      - 'max_keywords' (int, optional): The maximum
                                        number of keywords to return. Defaults to 5.
                                      - 'min_word_length' (int, optional): Minimum
                                        length a word must have to be considered
                                        a keyword. Defaults to 3.
                                      - 'stop_words' (list[str] or set[str], optional):
                                        A custom list or set of stop words to filter
                                        out. These will be added to the node's
                                        default stop words.

        Returns:
            List[str]: A list of extracted keywords, ordered by frequency
                       (most frequent first).

        Raises:
            ValueError: If the input 'data' is not a string, indicating an
                        incorrect data type for this node's operation.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"but received {type(data)}. Cannot extract keywords."
            )
            raise ValueError(f"Input data for '{self.node_name}' must be a string.")

        text = data.lower()

        # Retrieve parameters from context with sensible defaults
        max_keywords = context.get('max_keywords', 5)
        min_word_length = context.get('min_word_length', 3)
        custom_stop_words = context.get('stop_words')

        # Validate and combine default and custom stop words
        effective_stop_words = set(self._DEFAULT_STOP_WORDS)
        if custom_stop_words is not None:
            if isinstance(custom_stop_words, (list, set)):
                effective_stop_words.update({sw.lower() for sw in custom_stop_words})
            else:
                logger.warning(
                    f"[{self.node_name}] 'stop_words' in context should be a list or set. "
                    f"Received type '{type(custom_stop_words)}', ignoring custom stop words."
                )

        logger.debug(
            f"[{self.node_name}] Starting keyword extraction with parameters: "
            f"max_keywords={max_keywords}, min_word_length={min_word_length}, "
            f"custom_stop_words_provided={bool(custom_stop_words is not None)}"
        )

        # Use regex to find all word characters, which naturally handles punctuation
        words = re.findall(r'\b\w+\b', text)

        # Filter out stop words and words shorter than the minimum length
        filtered_words = [
            word for word in words
            if word not in effective_stop_words and len(word) >= min_word_length
        ]

        # Count the frequency of each filtered word
        word_counts = Counter(filtered_words)

        # Get the top N most common words as keywords
        keywords = [word for word, count in word_counts.most_common(max_keywords)]

        logger.info(
            f"[{self.node_name}] Successfully extracted {len(keywords)} keywords: {keywords}"
        )

        return keywords
