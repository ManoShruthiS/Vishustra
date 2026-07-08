import logging
import re
from typing import Any, Dict, List, Set

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text.
    It performs a basic text cleaning, tokenization, stop-word removal,
    and returns a list of unique keywords.
    """

    # A simple set of common English stop words for demonstration.
    # In a real-world scenario, this might be loaded from a more comprehensive
    # NLTK or SpaCy list, or configurable via node parameters.
    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "be", "was", "were",
        "and", "or", "but", "not", "for", "on", "in", "at", "of",
        "to", "from", "with", "by", "as", "it", "its", "i", "you",
        "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "our", "their", "this", "that", "these",
        "those", "what", "which", "who", "whom", "where", "when", "why",
        "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "only", "own", "same",
        "so", "than", "too", "very", "can", "will", "just", "don", "should",
        "now", "here", "there", "up", "down", "out", "off", "over", "under",
        "again", "further", "then", "once", "about", "could", "would",
        "had", "has", "have", "do", "does", "did", "am", "if", "else", "elsewhere"
    }
    _MIN_KEYWORD_LENGTH: int = 3 # Minimum length for a word to be considered a keyword

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        This method expects the `data` to be a string representing the text
        from which keywords are to be extracted. It performs the following steps:
        1. Converts the text to lowercase.
        2. Removes punctuation and numbers.
        3. Splits the text into individual words.
        4. Filters out common stop words defined in `_STOP_WORDS`.
        5. Filters out words shorter than `_MIN_KEYWORD_LENGTH`.
        6. Returns a list of unique extracted keywords, sorted alphabetically.

        Expected `data` type: `str`
        The `context` dictionary is available for future extensions or
        workflow-specific parameters, but is not used in this basic implementation.

        Args:
            data: The text string from which to extract keywords.
            context: A dictionary containing contextual information for processing,
                     passed through the Vishustra orchestration.

        Returns:
            A list of unique strings representing the extracted keywords.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Input data must be a string, "
                f"but received type: {type(data).__name__}"
            )
            raise TypeError(f"'{self.node_name}' node expects string input, got {type(data).__name__}")

        logger.info(f"[{self.node_name}] Starting keyword extraction for input data.")

        # Step 1: Normalize text to lowercase
        text = data.lower()

        # Step 2: Remove punctuation, numbers, and replace with space
        # This regex replaces any character that is not a lowercase letter or a space
        # with a single space.
        cleaned_text = re.sub(r'[^a-z\s]', ' ', text)

        # Step 3: Tokenize the cleaned text into words
        # and remove extra spaces that might result from punctuation removal
        words = cleaned_text.split()
        
        if not words:
            logger.warning(
                f"[{self.node_name}] No valid words found after cleaning and tokenization. "
                "Returning an empty list of keywords."
            )
            return [] # No words to process, return empty list

        # Step 4 & 5: Filter out stop words and short words
        keywords: Set[str] = set()
        for word in words:
            if word not in self._STOP_WORDS and len(word) >= self._MIN_KEYWORD_LENGTH:
                keywords.add(word)

        # Convert set to a sorted list for consistent and predictable output.
        extracted_keywords = sorted(list(keywords))

        logger.info(
            f"[{self.node_name}] Finished processing. "
            f"Extracted {len(extracted_keywords)} unique keywords."
        )
        return extracted_keywords
