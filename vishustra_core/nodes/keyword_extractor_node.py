import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node that extracts keywords from a given text.

    This node tokenizes the input text, filters out common stopwords and short words,
    and then ranks the remaining words by frequency to identify potential keywords.
    The number of keywords and minimum word length can be configured via the context.
    """

    def __init__(self):
        """
        Initializes the KeywordExtractorNode with a default set of stopwords.
        """
        super().__init__()
        self._stopwords: Set[str] = self._get_default_stopwords()
        logger.debug("KeywordExtractorNode initialized with default stopwords.")

    def _get_default_stopwords(self) -> Set[str]:
        """
        Provides a basic, English set of stopwords. In a production system, this
        might be loaded from a more comprehensive source or be configurable.
        """
        return {
            "a", "an", "the", "is", "are", "and", "or", "in", "on", "at", "for", "with", "to", "of",
            "it", "that", "this", "but", "by", "from", "as", "he", "she", "it", "they", "we", "you",
            "i", "me", "him", "her", "us", "them", "my", "your", "our", "their", "itself", "himself",
            "herself", "myself", "yourself", "ourselves", "themselves", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "not", "no", "yes", "can", "could", "would",
            "shall", "will", "may", "might", "must", "if", "then", "else", "when", "where", "why",
            "how", "what", "who", "whom", "whose", "which", "there", "here", "over", "under", "again",
            "further", "once", "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just",
            "don", "should", "now", "ve", "ll", "re", "m", "o", "d", "won", "didn", "doesn", "hadn",
            "hasn", "haven", "isn", "wasn", "weren", "wouldn", "couldn", "shouldn"
        }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract keywords.

        The extraction logic includes:
        1. Lowercasing the input text.
        2. Tokenizing the text into individual words.
        3. Filtering out words shorter than `min_word_length` (default: 3).
        4. Removing common stopwords.
        5. Counting the frequency of the remaining words.
        6. Returning the `num_keywords` most frequent words (default: 5).

        Configuration can be provided in the `context` dictionary:
        - `num_keywords`: int, the maximum number of keywords to return. Defaults to 5.
        - `min_word_length`: int, the minimum length for a word to be considered a keyword. Defaults to 3.

        Args:
            data (Any): The input data, expected to be a string (text).
            context (Dict[str, Any]): A dictionary containing runtime configuration
                                       and state for the current processing pipeline.

        Returns:
            List[str]: A list of extracted keywords, sorted by frequency (descending)
                       and then alphabetically (ascending) for ties.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for KeywordExtractorNode. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            raise TypeError("KeywordExtractorNode expects string data for processing.")

        text = data.lower()

        # Retrieve configuration from context with default values
        num_keywords: int = context.get("num_keywords", 5)
        min_word_length: int = context.get("min_word_length", 3)

        logger.info(
            f"Starting keyword extraction with: num_keywords={num_keywords}, "
            f"min_word_length={min_word_length}."
        )

        # Basic tokenization: find all sequences of word characters
        words = re.findall(r'\b\w+\b', text)

        filtered_words: List[str] = []
        for word in words:
            if len(word) >= min_word_length and word not in self._stopwords:
                filtered_words.append(word)

        # Count word frequencies
        word_counts: Dict[str, int] = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort keywords by frequency (descending) and then alphabetically (ascending) for ties
        sorted_keywords = sorted(
            word_counts.items(),
            key=lambda item: (-item[1], item[0])
        )

        # Extract the top N keywords
        keywords: List[str] = [word for word, count in sorted_keywords[:num_keywords]]

        logger.info(f"Successfully extracted {len(keywords)} keywords.")
        return keywords