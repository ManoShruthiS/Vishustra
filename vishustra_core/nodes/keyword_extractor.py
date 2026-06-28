import logging
import re
from typing import Any, Dict, List, Set, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# A small, configurable set of common English words to exclude for basic keyword extraction.
# In a production NLP scenario, this would be a much larger, language-specific list,
# potentially from a dedicated library like NLTK or spaCy.
_DEFAULT_COMMON_WORDS: Set[str] = {
    "the", "a", "an", "is", "of", "and", "in", "to", "it", "that", "on", "with", "for",
    "as", "by", "at", "from", "he", "she", "they", "we", "you", "i", "was", "were", "be",
    "has", "had", "do", "does", "did", "this", "that", "these", "those", "have", "not",
    "but", "or", "which", "when", "where", "how", "what", "who", "why", "then", "than",
    "there", "here", "them", "us", "me", "him", "her", "my", "your", "his", "their", "our",
    "its", "would", "could", "should", "will", "can", "may", "much", "more", "most", "also",
    "up", "down", "out", "about", "into", "over", "under", "all", "any", "some", "such",
    "no", "only", "very", "just", "so", "too", "new", "old", "good", "bad", "great",
    "many", "few", "other", "another", "first", "last", "long", "short", "even", "back",
    "well", "get", "go", "say", "see", "make", "take", "come", "know", "think", "look",
    "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave",
    "call", "need", "start", "run", "keep", "put", "mean", "begin", "show", "hear", "play",
    "talk", "read", "write", "move", "like", "love", "hate", "full", "empty", "high", "low",
    "through", "against", "between", "among", "without", "before", "after", "above", "below"
}


class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from text data.

    This node takes a string as input, processes it by tokenizing, filtering
    based on length and common words, and returns a list of unique keywords.
    It's designed for demonstration and basic functionality within the framework,
    simulating keyword extraction without external NLP libraries.
    """

    def __init__(
        self,
        max_keywords: int = 5,
        min_word_length: int = 3,
        exclude_common_words: bool = True,
        custom_common_words: Union[Set[str], List[str], None] = None
    ):
        """
        Initializes the KeywordExtractorNode.

        Args:
            max_keywords: The maximum number of keywords to return. Must be a positive integer.
            min_word_length: The minimum length a word must have to be considered a keyword.
                             Must be a positive integer.
            exclude_common_words: If True, common English stop words will be excluded.
            custom_common_words: An optional set or list of additional words to exclude.
                                 These will be converted to lowercase and added to the
                                 default common words if `exclude_common_words` is True.
        """
        if not isinstance(max_keywords, int) or max_keywords <= 0:
            raise ValueError(f"max_keywords must be a positive integer, got {max_keywords}")
        if not isinstance(min_word_length, int) or min_word_length <= 0:
            raise ValueError(f"min_word_length must be a positive integer, got {min_word_length}")
        if custom_common_words is not None and not isinstance(custom_common_words, (set, list)):
            raise TypeError("custom_common_words must be a set or list of strings, or None.")

        self._max_keywords = max_keywords
        self._min_word_length = min_word_length
        self._exclude_common_words = exclude_common_words

        self._common_words = set()
        if self._exclude_common_words:
            self._common_words.update(_DEFAULT_COMMON_WORDS)
            if custom_common_words:
                # Ensure custom words are also lowercased for consistent comparison
                self._common_words.update(word.lower() for word in custom_common_words if isinstance(word, str))

        logger.debug(
            f"KeywordExtractorNode initialized: max_keywords={self._max_keywords}, "
            f"min_word_length={self._min_word_length}, "
            f"exclude_common_words={self._exclude_common_words}. "
            f"Total common words for exclusion: {len(self._common_words) if self._exclude_common_words else 0}."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data based on configured rules.

        Args:
            data: The input text as a string from which to extract keywords.
            context: A dictionary containing contextual information, not directly used by this node.

        Returns:
            A list of unique keywords extracted from the text, sorted by length (descending)
            and then alphabetically (ascending), limited by `max_keywords`.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"KeywordExtractorNode expects string input for 'data', "
                f"but received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.info("Received empty or whitespace-only string; returning an empty list of keywords.")
            return []

        text = data.lower()
        # Use regex to find all sequences of word characters. This helps normalize punctuation.
        words = re.findall(r'\b\w+\b', text)

        candidate_keywords: List[str] = []
        for word in words:
            # Filter by minimum length
            if len(word) < self._min_word_length:
                logger.debug(f"Skipping word '{word}' (length {len(word)} < {self._min_word_length}).")
                continue

            # Filter out common words if configured
            if self._exclude_common_words and word in self._common_words:
                logger.debug(f"Skipping common word '{word}'.")
                continue

            candidate_keywords.append(word)

        # Remove duplicates and sort by length (descending) then alphabetically (ascending)
        unique_keywords = sorted(
            list(set(candidate_keywords)),
            key=lambda k: (-len(k), k)
        )

        # Truncate the list to the maximum number of keywords
        final_keywords = unique_keywords[:self._max_keywords]

        logger.info(
            f"Successfully extracted {len(final_keywords)} keywords from input text "
            f"(original length: {len(data)} characters). Keywords: {final_keywords}"
        )

        return final_keywords