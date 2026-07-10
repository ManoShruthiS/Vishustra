import logging
import re
from collections import Counter
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node responsible for extracting keywords from text data.

    This node preprocesses input text by converting it to lowercase, removing
    punctuation, and splitting it into individual words. It then filters these
    words based on a configurable list of stopwords and a minimum word length.
    Finally, it identifies and returns the most frequently occurring words
    as keywords, up to a specified maximum count.

    Configuration parameters can be passed via the 'context' dictionary, allowing
    flexible control over the extraction process.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string of text) to extract relevant keywords.

        Args:
            data: The input text from which keywords are to be extracted. Must be a string.
            context: A dictionary containing configuration parameters for keyword extraction.
                     Supported keys:
                     - 'max_keywords_to_extract' (int, optional): The maximum number of
                       keywords to return. Defaults to 5.
                     - 'min_word_length' (int, optional): The minimum length a word must
                       have to be considered a keyword. Defaults to 3.
                     - 'stopwords' (List[str], optional): A list of additional words to
                       exclude from keyword consideration. Merged with a default set.

        Returns:
            A list of strings, where each string is an extracted keyword, sorted
            by frequency in descending order.

        Raises:
            TypeError: If the input 'data' is not a string, indicating an invalid
                       data contract.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractorNode received invalid input type. Expected 'str', "
                f"but got '{type(data).__name__}'. Unable to process."
            )
            raise TypeError(
                f"KeywordExtractorNode requires string input, but received {type(data).__name__}."
            )

        # Handle empty or purely whitespace input gracefully
        if not data.strip():
            logger.info("KeywordExtractorNode received an empty or whitespace-only string. Returning an empty list of keywords.")
            return []

        # --- Retrieve configuration from context with sensible defaults ---
        max_keywords = context.get('max_keywords_to_extract', 5)
        min_word_length = context.get('min_word_length', 3)

        # Default common English stopwords. This list can be extended or overridden.
        default_stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for",
            "not", "in", "on", "at", "to", "from", "with", "as", "by", "of", "it",
            "its", "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
            "them", "my", "your", "our", "their", "this", "that", "these", "those",
            "what", "when", "where", "why", "how", "which", "who", "whom", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "can", "will",
            "would", "should", "could", "get", "go", "said", "say", "also", "many",
            "much", "more", "most", "some", "any", "no", "just", "only", "then",
            "now", "here", "there", "up", "down", "out", "off", "over", "under",
            "again", "further", "all", "both", "each", "few", "other", "such",
            "s", "t", "don", "shouldn", "nor", "too", "very", "etc"
        }
        custom_stopwords = set(context.get('stopwords', []))
        stopwords = default_stopwords.union(custom_stopwords)

        logger.debug(
            f"KeywordExtractorNode starting processing with "
            f"max_keywords={max_keywords}, min_word_length={min_word_length}."
        )

        # --- Text Preprocessing and Tokenization ---
        # Convert text to lowercase to standardize words
        text_lower = data.lower()
        # Use regex to find all sequences of word characters, effectively splitting
        # by spaces and removing punctuation.
        words = re.findall(r'\b\w+\b', text_lower)

        # --- Filtering words ---
        filtered_words = [
            word for word in words
            if word not in stopwords and len(word) >= min_word_length
        ]

        if not filtered_words:
            logger.info(
                "No significant words found after applying stopwords and "
                "minimum length filters. Returning an empty list."
            )
            return []

        # --- Keyword Extraction (Frequency Counting) ---
        word_counts = Counter(filtered_words)

        # Get the 'max_keywords' most common words
        # Counter.most_common returns a list of (word, count) tuples
        top_keywords_with_counts = word_counts.most_common(max_keywords)
        extracted_keywords = [keyword for keyword, count in top_keywords_with_counts]

        logger.debug(f"Successfully extracted {len(extracted_keywords)} keywords: {extracted_keywords}")
        return extracted_keywords