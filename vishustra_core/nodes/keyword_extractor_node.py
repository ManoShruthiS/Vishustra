import logging
import re
from typing import Any, Dict, List, Set, Union

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node that simulates extracting keywords from a given text.
    It processes a string input to identify and return a list of potential keywords
    based on basic text processing rules (e.g., filtering stop words, minimum length,
    and removing punctuation).
    """

    DEFAULT_STOP_WORDS = {
        "the", "a", "an", "is", "and", "or", "in", "of", "to", "for", "with", "on", "at", "by", "from",
        "it", "its", "as", "he", "she", "they", "we", "you", "i", "my", "your", "his", "her", "their",
        "our", "this", "that", "these", "those", "be", "been", "being", "was", "were", "are", "do",
        "does", "did", "not", "no", "yes", "but", "if", "then", "else", "when", "where", "why", "how",
        "what", "who", "whom", "which", "will", "would", "should", "could", "can", "may", "might",
        "must", "have", "has", "had", "just", "only", "also", "even", "much", "more", "most", "many",
        "some", "any", "all", "each", "every", "few", "other", "such", "so", "up", "down", "out",
        "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now"
    }
    DEFAULT_MIN_KEYWORD_LENGTH = 3

    def __init__(self,
                 stop_words: Union[Set[str], List[str], None] = None,
                 min_keyword_length: int = DEFAULT_MIN_KEYWORD_LENGTH):
        """
        Initializes the KeywordExtractorNode with configurable stop words and
        minimum keyword length.

        Args:
            stop_words: A set or list of words to exclude from keywords.
                        Defaults to a predefined set if None. All words are
                        converted to lowercase for case-insensitive matching.
            min_keyword_length: The minimum length a word must have to be considered
                                a keyword. Defaults to 3.
        """
        self._stop_words: Set[str] = set(word.lower() for word in stop_words) if stop_words else self.DEFAULT_STOP_WORDS
        self._min_keyword_length: int = min_keyword_length
        logger.debug(
            f"KeywordExtractorNode initialized with min_keyword_length={self._min_keyword_length} "
            f"and {len(self._stop_words)} stop words."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract keywords.

        The extraction process involves:
        1. Converting the text to lowercase.
        2. Splitting the text into words.
        3. Removing punctuation and non-alphabetic characters from each word.
        4. Filtering out words that are in the configured stop words list.
        5. Filtering out words shorter than the configured minimum length.
        6. Returning a sorted list of unique extracted keywords.

        Args:
            data: The input text as a string.
            context: A dictionary for shared context or state across nodes.
                     This node does not directly use the context for its core
                     keyword extraction logic, but it's available for potential
                     future extensions or debugging.

        Returns:
            A list of unique keywords extracted from the text, sorted alphabetically.

        Raises:
            ValueError: If the input data is not a string.
            Exception: For any unexpected errors encountered during the extraction process.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for KeywordExtractorNode '{self.node_name}'. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            raise ValueError(
                f"KeywordExtractorNode '{self.node_name}' requires 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        try:
            text = data.lower()
            # A more robust split that handles various delimiters and removes empty strings
            words = re.findall(r'\b\w+\b', text)

            extracted_keywords: Set[str] = set()

            for word in words:
                # Basic alphanumeric filter, `re.findall(r'\b\w+\b', text)` already handles this
                # but an extra layer ensures consistency if splitting logic changes.
                clean_word = ''.join(char for char in word if char.isalpha())

                if clean_word and \
                   len(clean_word) >= self._min_keyword_length and \
                   clean_word not in self._stop_words:
                    extracted_keywords.add(clean_word)

            sorted_keywords = sorted(list(extracted_keywords))
            logger.info(
                f"Successfully extracted {len(sorted_keywords)} keywords from input data "
                f"using node '{self.node_name}'."
            )
            return sorted_keywords
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during keyword extraction in node '{self.node_name}'. "
                f"Error: {e}"
            )
            # Re-raise the exception to propagate the error up the orchestration chain
            raise
