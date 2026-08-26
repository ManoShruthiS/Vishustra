import logging
import string
from typing import Any, Dict, List, Set

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class KeywordExtractor(BaseNode):
    """
    A Vishustra processing node that extracts keywords from text data.

    This node tokenizes input text, filters out common stop words and short words,
    and returns a unique list of potential keywords.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text.

        The `data` input is expected to be a string.
        The `context` can optionally provide:
        - 'stop_words' (Set[str] or List[str]): A collection of words to ignore.
          Defaults to common English stop words if not provided.
        - 'min_keyword_length' (int): Minimum length for a word to be considered a keyword.
          Defaults to 3.
        - 'remove_punctuation' (bool): Whether to remove punctuation before tokenizing.
          Defaults to True.
        - 'case_sensitive' (bool): Whether keyword extraction should be case-sensitive.
          Defaults to False.

        Args:
            data: The input text as a string.
            context: A dictionary containing configuration for keyword extraction.

        Returns:
            A list of unique keywords extracted from the text, preserving their original
            order of first appearance.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If context parameters are of an incorrect type or value.
        """
        if not isinstance(data, str):
            logger.error(
                "KeywordExtractor expects 'data' to be a string, but received type: %s",
                type(data).__name__,
            )
            raise TypeError(
                f"KeywordExtractor: Input 'data' must be a string, but got {type(data).__name__}"
            )

        # Default configurations
        default_stop_words: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for", "nor", "so", "yet",
            "in", "on", "at", "to", "from", "of", "with", "as", "by", "that", "this", "it", "he", "she",
            "we", "you", "they", "i", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
            "their", "what", "who", "when", "where", "why", "how", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "would", "could", "about",
            "above", "after", "again", "against", "ain", "am", "among", "an", "and", "any", "are", "aren",
            "around", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
            "but", "by", "can", "couldn", "d", "did", "didn", "do", "does", "doesn", "doing", "don", "down",
            "during", "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
            "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
            "into", "is", "isn", "it", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "more", "most",
            "mustn", "my", "myself", "needn", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only",
            "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "she",
            "should", "shouldn", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
            "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
            "until", "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which",
            "while", "who", "whom", "why", "will", "with", "won", "wouldn", "y", "you", "your", "yours", "yourself",
            "yourselves"
        }

        min_keyword_length: int = context.get("min_keyword_length", 3)
        remove_punctuation: bool = context.get("remove_punctuation", True)
        case_sensitive: bool = context.get("case_sensitive", False)

        # Validate context parameters
        if not isinstance(min_keyword_length, int) or min_keyword_length < 0:
            logger.error(
                "KeywordExtractor: 'min_keyword_length' must be a non-negative integer. Got: %s",
                min_keyword_length,
            )
            raise ValueError("'min_keyword_length' must be a non-negative integer.")
        if not isinstance(remove_punctuation, bool):
            logger.error(
                "KeywordExtractor: 'remove_punctuation' must be a boolean. Got: %s",
                remove_punctuation,
            )
            raise ValueError("'remove_punctuation' must be a boolean.")
        if not isinstance(case_sensitive, bool):
            logger.error(
                "KeywordExtractor: 'case_sensitive' must be a boolean. Got: %s",
                case_sensitive,
            )
            raise ValueError("'case_sensitive' must be a boolean.")

        # Get stop words from context or use default
        stop_words_raw = context.get("stop_words", default_stop_words)
        if not isinstance(stop_words_raw, (set, list, tuple)):
            logger.error(
                "KeywordExtractor: 'stop_words' must be a set, list, or tuple. Got: %s",
                type(stop_words_raw).__name__,
            )
            raise ValueError(
                f"'stop_words' must be a set, list, or tuple. Got: {type(stop_words_raw).__name__}"
            )
        try:
            stop_words: Set[str] = set(stop_words_raw)
            if isinstance(stop_words_raw, (list, tuple)):
                logger.debug("KeywordExtractor: 'stop_words' provided as a list/tuple. Converted to set.")
        except TypeError as e:
            logger.error("KeywordExtractor: Could not convert 'stop_words' to a set. Error: %s", e)
            raise ValueError(f"'stop_words' must contain hashable elements. Error: {e}")

        processed_text = data
        if not case_sensitive:
            processed_text = processed_text.lower()
            # Ensure all stop words are also lowercased for case-insensitive comparison
            stop_words = {word.lower() for word in stop_words}

        if remove_punctuation:
            # Create a translation table to remove all punctuation characters
            translator = str.maketrans('', '', string.punctuation)
            processed_text = processed_text.translate(translator)

        # Split text into words, handling multiple spaces
        words = processed_text.split()
        extracted_keywords: List[str] = []
        seen_keywords: Set[str] = set()

        for word in words:
            # Check if word is not empty, not a stop word, and meets minimum length
            if word and word not in stop_words and len(word) >= min_keyword_length:
                if word not in seen_keywords:
                    extracted_keywords.append(word)
                    seen_keywords.add(word)

        logger.info(
            "KeywordExtractor processed text and extracted %d unique keywords.",
            len(extracted_keywords),
        )
        return extracted_keywords