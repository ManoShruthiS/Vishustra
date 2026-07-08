import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text.

    This node processes the input text by tokenizing it, converting words
    to lowercase, removing punctuation, filtering out common stop words,
    and enforcing a minimum word length to identify potential keywords.
    """

    # A simple, illustrative set of stop words. In a real-world application,
    # this would typically be more comprehensive, potentially loaded from
    # an external resource, and configurable.
    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "else", "for", "with", "at", "from",
        "of", "on", "in", "to", "it", "its", "me", "my", "you", "your", "he",
        "she", "his", "her", "we", "our", "they", "their", "this", "that",
        "these", "those", "can", "will", "may", "must", "have", "has", "had",
        "do", "does", "did", "not", "no", "yes", "so", "as", "about", "above",
        "below", "between", "each", "other", "such", "than", "too", "very",
        "s", "t", "d", "ll", "m", "re", "ve", "y", "am", "i", "get", "go",
        "just", "make", "know", "see", "think", "take", "come", "want", "look",
        "say", "tell", "give", "find", "use", "work", "would", "could", "should",
        "also", "much", "many", "even", "much"
    }
    _DEFAULT_MIN_WORD_LENGTH: int = 3

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        This method expects the `data` parameter to be a string containing
        the text from which keywords need to be extracted. It performs the
        following steps:
        1. Validates that the input `data` is a string.
        2. Retrieves configuration parameters (stop words, minimum word length)
           from the `context` dictionary or uses default values.
        3. Tokenizes the text into words, converting them to lowercase.
        4. Filters out words that are in the configured stop word list.
        5. Filters out words that do not meet the minimum word length.
        6. Returns a unique, sorted list of the remaining words as keywords.

        Configuration can be provided via the `context` dictionary under the
        key `keyword_extractor_config`. Supported sub-keys within this config:
        - `stop_words` (Set[str]): An optional set of words to ignore. If provided,
                                   this set completely replaces the default stop words.
        - `min_word_length` (int): An optional integer specifying the minimum
                                   length for a word to be considered a keyword.

        Args:
            data: The input text from which keywords will be extracted.
                  Expected type is `str`.
            context: A dictionary containing contextual information and
                     optional configuration for the node.

        Returns:
            A `List[str]` where each string is an extracted keyword.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting keyword extraction process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Raising ValueError."
            )
            raise ValueError(
                f"{self.node_name} expects input 'data' to be a string, "
                f"but got {type(data).__name__}."
            )

        # Retrieve configuration from context or fall back to defaults
        config = context.get("keyword_extractor_config", {})
        stop_words = config.get("stop_words", self._DEFAULT_STOP_WORDS)
        min_word_length = config.get("min_word_length", self._DEFAULT_MIN_WORD_LENGTH)

        logger.debug(
            f"[{self.node_name}] Configuration: min_word_length={min_word_length}, "
            f"stop_words_count={len(stop_words)}."
        )

        extracted_keywords: Set[str] = set()
        # Use regex to find all alphanumeric word boundaries, converting to lowercase
        # This handles various punctuation and spacing scenarios robustly.
        words = re.findall(r'\b\w+\b', data.lower())

        for word in words:
            # Further strip to ensure no accidental whitespace, though regex is usually clean
            cleaned_word = word.strip()

            if len(cleaned_word) >= min_word_length and cleaned_word not in stop_words:
                extracted_keywords.add(cleaned_word)

        # Convert the set of unique keywords to a sorted list for consistent output
        result = sorted(list(extracted_keywords))

        logger.info(
            f"[{self.node_name}] Successfully extracted {len(result)} unique keywords."
        )
        logger.debug(f"[{self.node_name}] Extracted keywords: {result}")
        return result