
import re
import logging
from typing import Any, Dict, Set, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra node designed to extract keywords from a given text string.

    This node performs basic text processing steps: lowercasing, punctuation
    removal, and tokenization. It then filters words based on a configurable
    set of stop words and a minimum word length to identify and return a
    set of prominent keywords.
    """

    DEFAULT_STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for", "nor", "this", "that",
        "it", "s", "ve", "ll", "re", "m", "t", "d", "won", "shan", "don", "not", "off", "to", "in", "on",
        "at", "from", "by", "with", "about", "as", "he", "she", "it", "they", "we", "you", "i", "me", "him",
        "her", "us", "them", "my", "your", "his", "hers", "its", "our", "their", "so", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
        "must", "etc", "such", "all", "any", "both", "each", "few", "more", "most", "other", "some", "what",
        "which", "who", "whom", "yours", "myself", "yourself", "himself", "herself", "itself", "ourselves",
        "yourselves", "themselves", "many", "much", "no", "just", "only", "own", "same", "too", "very",
        "through", "down", "out", "up", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should",
        "now"
    }

    def __init__(self, stop_words: Optional[Set[str]] = None, min_word_length: int = 3):
        """
        Initializes the KeywordExtractor node with configurable stop words and minimum word length.

        Args:
            stop_words (Optional[Set[str]]): An optional set of custom stop words. If None,
                                              a robust default set will be used.
            min_word_length (int): The minimum character length a word must have to be
                                   considered a keyword. Words shorter than this will be filtered out.
                                   Must be at least 1.
        """
        self._stop_words = stop_words if stop_words is not None else self.DEFAULT_STOP_WORDS
        if not isinstance(self._stop_words, set):
            try:
                self._stop_words = set(self._stop_words)
                logger.warning(f"Provided 'stop_words' was not a set. Converted to set. Type: {type(stop_words)}")
            except TypeError:
                logger.error(f"Provided 'stop_words' ({type(stop_words)}) could not be converted to a set. Using default stop words.")
                self._stop_words = self.DEFAULT_STOP_WORDS

        # Ensure min_word_length is a positive integer
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(f"Invalid min_word_length provided: {min_word_length}. Setting to default (3).")
            self._min_word_length = 3
        else:
            self._min_word_length = min_word_length

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> Set[str]:
        """
        Processes the input data to extract a set of unique keywords.

        The `context` dictionary can be used to dynamically override the
        `stop_words` and `min_word_length` for a specific processing call.

        Args:
            data (Any): The input data, expected to be a string or convertible to a string,
                        containing the text from which to extract keywords.
            context (Dict[str, Any]): A dictionary providing additional context or
                                       configuration for this processing call.
                                       Accepted keys:
                                       - 'stop_words' (Set[str]): Overrides the node's
                                                                  configured stop words.
                                       - 'min_word_length' (int): Overrides the node's
                                                                  configured minimum word length.

        Returns:
            Set[str]: A set of unique keywords extracted from the text.
                      Returns an empty set if input data is invalid, empty, or no keywords are found.

        Raises:
            TypeError: If the input `data` cannot be converted to a string.
        """
        if not isinstance(data, str):
            try:
                text = str(data)
                logger.info(f"Input data of type {type(data)} converted to string for processing.")
            except Exception as e:
                logger.error(f"Failed to convert input data of type {type(data)} to string: {e}")
                raise TypeError(
                    f"Input data must be string-like, but received {type(data)} and conversion failed."
                ) from e
        else:
            text = data

        if not text.strip():
            logger.warning("Received empty or whitespace-only text for keyword extraction. Returning empty set.")
            return set()

        # Determine stop words for this specific call, prioritizing context
        current_stop_words = context.get('stop_words', self._stop_words)
        if not isinstance(current_stop_words, set):
            try:
                current_stop_words = set(current_stop_words)
            except TypeError:
                logger.warning(
                    f"Context 'stop_words' is not a set and could not be converted. "
                    f"Using node's configured stop words. Type: {type(context.get('stop_words'))}"
                )
                current_stop_words = self._stop_words

        # Determine min word length for this specific call, prioritizing context
        current_min_word_length = context.get('min_word_length', self._min_word_length)
        if not isinstance(current_min_word_length, int) or current_min_word_length < 1:
            logger.warning(
                f"Context 'min_word_length' is invalid ({current_min_word_length}). "
                f"Using node's configured min_word_length: {self._min_word_length}."
            )
            current_min_word_length = self._min_word_length
        current_min_word_length = max(1, current_min_word_length) # Ensure it's always at least 1

        # Basic text processing pipeline
        lower_text = text.lower()
        # Remove non-alphanumeric characters (excluding spaces)
        cleaned_text = re.sub(r'[^a-z0-9\s]', '', lower_text)
        # Split into words (handles multiple spaces after cleaning)
        words = cleaned_text.split()

        extracted_keywords = set()
        for word in words:
            if word and len(word) >= current_min_word_length and word not in current_stop_words:
                extracted_keywords.add(word)

        logger.info(f"Successfully extracted {len(extracted_keywords)} keywords from the text.")
        return extracted_keywords
