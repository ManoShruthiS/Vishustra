import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra processing node that extracts potential keywords from a given text.

    This node performs a basic keyword extraction by:
    - Lowercasing the input text.
    - Tokenizing the text into individual words.
    - Removing common English stopwords and short words (less than 3 characters).
    - Filtering out punctuation.
    - Returning a list of unique, sorted keywords.
    """

    _stopwords: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "this", "that", "these", "those",
        "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
        "for", "to", "in", "on", "at", "by", "with", "from", "of", "about", "as",
        "it", "he", "she", "we", "you", "they", "i", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
        "should", "may", "might", "must", "be", "been", "being", "not", "no", "yes",
        "get", "go", "make", "take", "come", "see", "know", "think", "say", "tell",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "up", "down", "out", "in", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "nor",
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now", "" # Include empty string for robustness
    }
    _min_word_length: int = 3 # Minimum length for a word to be considered a keyword

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data (Any): The input data, expected to be a string (text document).
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by this
                                       node for keyword extraction logic, but available.

        Returns:
            List[str]: A list of unique keywords extracted from the text, sorted alphabetically.
                       Returns an empty list if data is not a string, is empty, or no keywords are found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'. Returning an empty list."
            )
            return []

        if not data.strip():
            logger.debug(
                f"[{self.node_name}] Received empty or whitespace-only string. "
                "Returning an empty list of keywords."
            )
            return []

        # Convert to lowercase and tokenize, removing punctuation
        text_lower = data.lower()
        # Use regex to find word characters, robustly handles various punctuation
        words = re.findall(r'\b\w+\b', text_lower)

        # Filter out stopwords and short words
        keywords_set: Set[str] = set()
        for word in words:
            if (
                word not in self._stopwords and
                len(word) >= self._min_word_length
            ):
                keywords_set.add(word)

        # Convert set to sorted list for consistent output
        extracted_keywords = sorted(list(keywords_set))

        logger.info(
            f"[{self.node_name}] Successfully extracted {len(extracted_keywords)} keywords "
            f"from input of length {len(data)}."
        )

        return extracted_keywords