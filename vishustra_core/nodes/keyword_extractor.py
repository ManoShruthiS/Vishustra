import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra node designed to extract keywords from text data.

    This node simulates keyword extraction by performing basic text processing:
    tokenization, lowercasing, stop word removal, and filtering out
    non-alphabetic tokens. It serves as a foundational component for
    text analysis workflows within Vishustra.
    """

    # A pre-defined set of common English stopwords.
    # For production use, this set would typically be loaded from a configuration
    # or an external linguistic resource, and potentially expanded or customized.
    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "by", "for", "with", "from", "to", "and", "or",
        "but", "as", "if", "not", "no", "yes", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
        "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
        "do", "does", "did", "don", "should", "s", "t", "can", "will", "would", "shall", "could",
        "might", "must", "have", "has", "had", "just", "about", "above", "after",
        "again", "all", "any", "before", "below", "between", "both", "each", "few",
        "more", "most", "other", "some", "such", "than", "then", "there", "down", "out",
        "up", "off", "only", "own", "same", "so", "too", "very", "now", "here", "there",
        "where", "when", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
        "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
        "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        This method expects the input `data` to be a string of text. It
        tokenizes the text, converts tokens to lowercase, removes common
        stopwords, and filters out non-alphabetic tokens. The result is a
        sorted list of unique keywords.

        Args:
            data (Any): The input data, expected to be a string containing
                        the text from which keywords are to be extracted.
            context (Dict[str, Any]): A dictionary providing contextual information.
                                      Currently not used by this node but part of
                                      the `BaseNode` interface.

        Returns:
            List[str]: A sorted list of unique extracted keywords. Returns an
                       empty list if the input is invalid or no keywords are found.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', got '%s'. Data: %s",
                self.node_name, type(data).__name__, str(data)[:100] # Truncate for log
            )
            raise TypeError(
                f"{self.node_name} expected a string input, but received {type(data).__name__}."
            )

        text = data.lower()
        logger.debug("[%s] Starting keyword extraction for text (truncated to 100 chars): '%s'...",
                     self.node_name, text[:100])

        # Tokenize the text by finding sequences of word characters.
        # This approach handles basic punctuation and whitespace.
        # For more advanced NLP tasks, a dedicated tokenizer (e.g., NLTK's word_tokenize)
        # would be preferred.
        words = re.findall(r'\b\w+\b', text)

        extracted_keywords: List[str] = []
        for word in words:
            # Ensure the word is not empty and is not a stop word.
            if word and word not in self._STOP_WORDS:
                extracted_keywords.append(word)

        # Convert to a set to ensure uniqueness, then back to a sorted list
        # for consistent output order.
        unique_keywords = sorted(list(set(extracted_keywords)))

        logger.info("[%s] Successfully extracted %d unique keywords.",
                    self.node_name, len(unique_keywords))
        logger.debug("[%s] Keywords found: %s", self.node_name, unique_keywords)

        return unique_keywords
