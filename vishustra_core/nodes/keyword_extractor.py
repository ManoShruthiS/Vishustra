import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# A predefined set of common English stop words for demonstration purposes.
# In a real-world scenario, this might be loaded from a more comprehensive NLP library or configuration.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
    "for", "to", "with", "at", "by", "from", "on", "off", "up", "down", "in", "out",
    "this", "that", "these", "those", "he", "she", "it", "we", "you", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
    "i", "you", "we", "they", "it", "not", "no", "yes", "can", "will", "would", "should", "could",
    "do", "does", "did", "don't", "doesn't", "didn't", "have", "has", "had", "haven't", "hasn't", "hadn't",
    "shall", "may", "might", "must", "much", "many", "just", "only", "such", "too", "very", "also", "get", "go",
    "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}


class KeywordExtractor(BaseNode):
    """
    A processing node that extracts keywords from a given text string.

    This node tokenizes the input text, converts words to lowercase,
    removes punctuation, filters out common stop words, and returns
    a sorted list of unique keywords.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data: The input text as a string from which to extract keywords.
            context: A dictionary containing contextual information
                     (not directly used by this node but required by BaseNode).

        Returns:
            A sorted list of unique keyword strings.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(
                f"KeywordExtractor expects 'data' to be a string, but received {type(data).__name__}."
            )

        text = data.strip()
        if not text:
            logger.warning(f"[{self.node_name}] Received an empty or whitespace-only string. No keywords to extract.")
            return []

        # Tokenize the text by splitting on non-alphanumeric characters, convert to lowercase
        # and filter out empty strings resulting from multiple delimiters.
        words = [
            word.lower() for word in re.findall(r'\b\w+\b', text)
            if word # ensure word is not empty after regex
        ]

        extracted_keywords: Set[str] = set()
        for word in words:
            # Basic cleaning: remove any remaining non-alphanumeric characters at word boundaries
            cleaned_word = re.sub(r'^\W+|\W+$', '', word)
            if cleaned_word and cleaned_word not in _STOP_WORDS:
                extracted_keywords.add(cleaned_word)

        # Sort the keywords for consistent output
        keywords_list = sorted(list(extracted_keywords))

        logger.debug(f"[{self.node_name}] Successfully extracted {len(keywords_list)} keywords.")
        return keywords_list