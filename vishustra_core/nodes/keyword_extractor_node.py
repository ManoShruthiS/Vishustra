import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set

# Assume BaseNode is available from this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node that extracts keywords from a given text string.

    This node tokenizes the input text, filters out common stop words and
    short words, and then identifies the most frequent remaining words
    as keywords. Configuration such as the number of keywords to extract
    and custom stop words can be provided via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data (Any): The input data, expected to be a string containing the text.
            context (Dict[str, Any]): A dictionary containing runtime configuration.
                                       Expected keys:
                                       - 'top_n' (int, optional): The maximum number of keywords to return. Defaults to 5.
                                       - 'min_word_length' (int, optional): Minimum length for a word to be considered. Defaults to 3.
                                       - 'stop_words' (List[str], optional): A custom list of stop words to exclude.
                                                                           Defaults to a common English stop word list.

        Returns:
            List[str]: A list of extracted keywords, sorted by frequency in descending order.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        logger.info(f"Node '{self.node_name}' initiated processing.")

        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for '{self.node_name}'. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string. "
                f"Got '{type(data).__name__}'."
            )

        text_content = data.lower()

        # Configuration from context, with sensible defaults
        top_n: int = context.get('top_n', 5)
        min_word_length: int = context.get('min_word_length', 3)

        # Default stop words. Can be overridden by context.
        default_stop_words: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "to", "of", "and", "or", "but", "if", "then", "else", "for", "with",
            "on", "at", "by", "from", "into", "during", "in", "out", "up", "down",
            "he", "she", "it", "we", "you", "they", "i", "me", "him", "her", "us", "them",
            "my", "your", "his", "our", "their", "this", "that", "these", "those",
            "have", "has", "had", "do", "does", "did", "not", "no", "yes", "can", "will",
            "would", "should", "could", "get", "go", "say", "see", "make", "know", "take",
            "about", "above", "after", "again", "all", "any", "before", "below", "between",
            "both", "each", "few", "more", "most", "other", "some", "such", "only", "own",
            "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
            "etc", "etc."
        }
        stop_words: Set[str] = set(context.get('stop_words', default_stop_words))

        # Tokenize and clean words: remove punctuation, filter by length and stop words.
        # Using regex to split by non-alphanumeric characters, ensuring words are clean.
        words = re.findall(r'\b\w+\b', text_content)
        
        filtered_words = [
            word for word in words
            if len(word) >= min_word_length and word not in stop_words
        ]

        if not filtered_words:
            logger.warning(
                f"Node '{self.node_name}' found no significant keywords "
                "after filtering, returning an empty list."
            )
            return []

        # Count word frequencies
        word_counts: Counter[str] = Counter(filtered_words)

        # Get the top N most common words
        extracted_keywords: List[str] = [
            word for word, count in word_counts.most_common(top_n)
        ]

        logger.info(
            f"Node '{self.node_name}' successfully extracted {len(extracted_keywords)} keywords."
        )
        return extracted_keywords