from vishustra_core.nodes.base_node import BaseNode
import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract keywords from text input.

    This node performs basic text cleaning, tokenization, and filtering
    based on word length and a configurable list of stopwords. It's intended
    to provide a preliminary set of keywords from a given document or utterance.
    """

    def __init__(self, min_word_length: int = 3, max_keywords: int = 10, custom_stopwords: List[str] = None):
        """
        Initializes the KeywordExtractorNode with specific configuration for extraction.

        Args:
            min_word_length (int): The minimum character length a word must have
                                   to be considered a keyword. Defaults to 3.
            max_keywords (int): The maximum number of keywords to return. Defaults to 10.
            custom_stopwords (List[str], optional): A list of additional words to exclude
                                                    during keyword extraction. If None,
                                                    only default stopwords are used.
        """
        if not isinstance(min_word_length, int) or min_word_length < 1:
            raise ValueError("min_word_length must be a positive integer.")
        if not isinstance(max_keywords, int) or max_keywords < 1:
            raise ValueError("max_keywords must be a positive integer.")
        if custom_stopwords is not None and not isinstance(custom_stopwords, list):
            raise TypeError("custom_stopwords must be a list of strings or None.")

        self._min_word_length = min_word_length
        self._max_keywords = max_keywords
        self._stopwords: Set[str] = self._build_stopwords_set(custom_stopwords)
        logger.debug(f"KeywordExtractorNode initialized: min_word_length={self._min_word_length}, "
                     f"max_keywords={self._max_keywords}, stopwords_count={len(self._stopwords)}")

    def _build_stopwords_set(self, custom_stopwords: List[str] | None) -> Set[str]:
        """
        Combines default and custom stopwords into a single set for efficient lookup.
        """
        default_stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
            "for", "to", "with", "without", "at", "by", "from", "into", "on", "off",
            "in", "out", "up", "down", "over", "under", "about", "above", "below",
            "before", "after", "again", "further", "once", "here", "there", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
            "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
            "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
            "shouldn", "wasn", "weren", "won", "wouldn"
        }
        if custom_stopwords:
            default_stopwords.update(word.lower() for word in custom_stopwords)
        return default_stopwords

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data (Any): The input data, expected to be a string from which
                        keywords will be extracted.
            context (Dict[str, Any]): A dictionary providing additional runtime context.
                                       This node does not currently use the context for
                                       configuration but future extensions could.

        Returns:
            List[str]: A list of unique keywords extracted from the text, sorted
                       alphabetically. The list will not exceed `max_keywords`.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping whitespace.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input type for KeywordExtractorNode. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"KeywordExtractorNode expects string data, but received {type(data).__name__}")

        text = data.strip()
        if not text:
            logger.warning("Received an empty string for keyword extraction. Returning an empty list.")
            return []

        logger.debug(f"Initiating keyword extraction for text of length {len(text)}.")

        # Convert to lowercase for consistent processing
        text = text.lower()

        # Remove non-alphanumeric characters (keeping spaces), then remove extra spaces
        cleaned_text = re.sub(r'[^a-z\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        if not cleaned_text:
            logger.debug("Text became empty after cleaning. No keywords to extract.")
            return []

        words = cleaned_text.split()
        
        extracted_keywords_set: Set[str] = set()
        
        for word in words:
            if (word not in self._stopwords and
                    len(word) >= self._min_word_length):
                extracted_keywords_set.add(word)
                if len(extracted_keywords_set) >= self._max_keywords:
                    logger.debug(f"Reached maximum of {self._max_keywords} keywords. Stopping extraction.")
                    break

        # Convert set to list and sort for deterministic output
        result_keywords = sorted(list(extracted_keywords_set))

        logger.info(f"Successfully extracted {len(result_keywords)} keywords.")
        logger.debug(f"Extracted keywords: {result_keywords}")
        
        return result_keywords[:self._max_keywords] # Ensure max_keywords limit is strictly enforced if set was not limited exactly
