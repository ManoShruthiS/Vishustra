import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract keywords from textual input.

    This node performs a series of text processing steps:
    1. Input validation to ensure the data is a string.
    2. Normalization: converts text to lowercase.
    3. Cleaning: removes punctuation and special characters.
    4. Tokenization: splits the cleaned text into individual words.
    5. Filtering: removes common English stop words and words below a minimum length.
    6. Uniqueness: ensures each extracted keyword is unique.

    The extracted keywords are returned as a sorted list of strings.
    """

    # A predefined set of common English stop words.
    # In a production system, this could be loaded from a configuration,
    # a specialized NLP library, or dynamically provided via the context.
    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "and", "or", "but",
        "for", "with", "to", "from", "in", "on", "at", "it", "this", "that",
        "of", "be", "have", "do", "you", "me", "he", "she", "we", "they",
        "i", "my", "your", "his", "her", "its", "our", "their", "them",
        "us", "him", "her", "itself", "myself", "yourself", "ourselves",
        "yourselves", "himself", "herself", "themselves", "what", "which",
        "who", "whom", "where", "when", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "would", "could",
        "get", "go", "say", "see", "make", "take", "come", "know", "think",
        "look", "want", "give", "use", "find", "tell", "ask", "work", "seem",
        "feel", "try", "leave", "call", "may", "might", "must", "many"
    }
    _MIN_KEYWORD_LENGTH: int = 3  # Minimum character length for a word to be considered a keyword.

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        Args:
            data (Any): The primary input data, expected to be a string
                        containing the text from which to extract keywords.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      or configuration. While not strictly used for
                                      dynamic configuration in this initial version,
                                      it serves as a placeholder for future extensions
                                      (e.g., custom stop words, min length).

        Returns:
            List[str]: A sorted list of unique keywords extracted from the input text.
                       Returns an empty list if no keywords are found or if the input
                       text is empty after cleaning.

        Raises:
            TypeError: If the input 'data' is not a string, indicating an invalid
                       data type for this node's operation.
        """
        logger.debug(f"[{self.node_name}] Initiating keyword extraction process.")

        # Validate input data type
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type received. Expected 'str', "
                f"but got '{type(data).__name__}'."
            )
            raise TypeError(
                f"KeywordExtractorNode requires string input. "
                f"Received type: {type(data).__name__}."
            )

        # Strip leading/trailing whitespace and check for empty content
        text = data.strip()
        if not text:
            logger.info(f"[{self.node_name}] Received empty or whitespace-only string. Returning empty keyword list.")
            return []

        # 1. Normalize text to lowercase
        normalized_text = text.lower()
        logger.debug(f"[{self.node_name}] Text normalized to lowercase.")

        # 2. Clean text: remove non-alphanumeric characters (except spaces)
        # This regex replaces any character that is not a letter, a number, or a space with a single space.
        cleaned_text = re.sub(r'[^a-z0-9\s]', ' ', normalized_text)
        logger.debug(f"[{self.node_name}] Text cleaned from special characters.")

        # 3. Tokenize: split into words and filter
        words = cleaned_text.split()
        extracted_keywords = set()

        for word in words:
            # Filter criteria: word must not be empty, meet minimum length, and not be a stop word
            if word and len(word) >= self._MIN_KEYWORD_LENGTH and word not in self._STOP_WORDS:
                extracted_keywords.add(word)

        # Convert the set to a sorted list for consistent output
        result = sorted(list(extracted_keywords))

        logger.info(f"[{self.node_name}] Successfully extracted {len(result)} unique keywords.")
        logger.debug(f"[{self.node_name}] Extracted keywords: {result}")
        return result
