import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract keywords from textual data.

    This node accepts a string as input and produces a ranked list of potential keywords.
    The keyword extraction process is configurable through the 'context' dictionary,
    allowing customization of common stopwords, the minimum length for a word to
    be considered a keyword, and the maximum number of keywords to return.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Executes the keyword extraction process on the input data.

        The method expects 'data' to be a string. It tokenizes the text, filters
        out stopwords and short words, then ranks the remaining words by frequency
        to identify potential keywords.

        Configuration parameters can be provided via the 'context' dictionary:
        - 'stopwords' (List[str], optional): A collection of words to ignore during
          extraction. Defaults to a comprehensive list of common English words.
        - 'min_word_length' (int, optional): The shortest allowed length for a word
          to be considered a keyword. Defaults to 3 characters.
        - 'max_keywords' (int, optional): The maximum number of top-ranked keywords
          to be returned. Defaults to 10.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        from which keywords are to be extracted.
            context (Dict[str, Any]): A dictionary providing runtime configuration
                                      settings for the keyword extraction logic.

        Returns:
            List[str]: An ordered list of extracted keywords, with higher-frequency
                       keywords appearing earlier.

        Raises:
            ValueError: If the input 'data' is not of type 'str'.
        """
        logger.info("KeywordExtractorNode: Initiating keyword extraction for new data.")

        if not isinstance(data, str):
            logger.error(
                "KeywordExtractorNode: Invalid input data type. Expected 'str', received '%s'.",
                type(data).__name__
            )
            raise ValueError(
                f"KeywordExtractorNode requires 'data' to be a string, but received '{type(data).__name__}'."
            )

        text = data.lower()

        # Define default stopwords if not provided in context
        default_stopwords: List[str] = [
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for",
            "nor", "so", "yet", "at", "by", "in", "on", "of", "to", "from", "with",
            "it", "its", "he", "she", "i", "we", "you", "they", "them", "us", "him",
            "her", "this", "that", "these", "those", "can", "will", "may", "would",
            "should", "could", "be", "been", "being", "do", "does", "did", "not",
            "has", "have", "had", "as", "about", "above", "after", "again", "all",
            "any", "because", "before", "both", "down", "during", "each", "few",
            "more", "most", "other", "some", "such", "than", "then", "there", "up",
            "very", "what", "when", "where", "which", "who", "whom", "why", "how",
            "just", "too", "own", "out", "off", "on", "only", "once", "over", "under",
            "while", "wherefore", "whereupon", "wherever", "whence", "wherein",
            "whithersoever", "whoever", "whose", "yours", "yourself", "yourselves",
            "we'll", "we'd", "we've", "we're", "i'll", "i'd", "i've", "i'm", "you'll",
            "you'd", "you've", "you're", "he'll", "he'd", "he's", "she'll", "she'd",
            "she's", "it'll", "it'd", "it's", "they'll", "they'd", "they've", "they're",
            "don't", "didn't", "doesn't", "haven't", "hasn't", "hadn't", "won't",
            "wouldn't", "can't", "couldn't", "shouldn't", "mustn't", "needn't"
        ]
        stopwords: Set[str] = set(context.get("stopwords", default_stopwords))
        min_word_length: int = context.get("min_word_length", 3)
        max_keywords: int = context.get("max_keywords", 10)

        logger.debug(
            "KeywordExtractorNode: Configuration - min_word_length=%d, max_keywords=%d, num_stopwords=%d",
            min_word_length, max_keywords, len(stopwords)
        )

        # Basic tokenization: split by non-alphabetic characters and convert to lowercase
        # Use a more robust regex for word splitting to handle various punctuation
        words: List[str] = re.findall(r'\b[a-z]+\b', text)

        # Filter words based on length and stopwords
        filtered_words: List[str] = []
        for word in words:
            if len(word) >= min_word_length and word not in stopwords:
                filtered_words.append(word)

        # Count frequencies of the filtered words
        word_counts: Dict[str, int] = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort words by frequency in descending order
        sorted_word_items = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

        # Extract the top N keywords
        extracted_keywords: List[str] = [word for word, count in sorted_word_items[:max_keywords]]

        logger.info(
            "KeywordExtractorNode: Successfully identified %d keywords (out of a max of %d requested).",
            len(extracted_keywords), max_keywords
        )
        return extracted_keywords