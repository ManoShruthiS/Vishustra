import logging
import re
from collections import Counter
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the Python path
# For local development/testing, you might mock this or ensure the path is set.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node that extracts keywords from a given text.

    It tokenizes the input text, filters words based on length,
    counts their frequency, and returns the top N most frequent words
    as keywords.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input data.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing execution context.
                                      Expected keys:
                                      - 'min_keyword_length' (int, optional): Minimum
                                        length for a word to be considered a keyword.
                                        Defaults to 3.
                                      - 'top_n_keywords' (int, optional): The maximum
                                        number of top keywords to return. Defaults to 10.

        Returns:
            List[str]: A list of extracted keywords, sorted by frequency (descending).

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'min_keyword_length' or 'top_n_keywords' in context are invalid.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for KeywordExtractor: Expected str, got {type(data).__name__}")
            raise TypeError(f"KeywordExtractor expects 'data' to be a string, but received {type(data).__name__}.")

        min_keyword_length: int = context.get('min_keyword_length', 3)
        top_n_keywords: int = context.get('top_n_keywords', 10)

        # Validate context parameters
        if not isinstance(min_keyword_length, int) or min_keyword_length <= 0:
            logger.error(f"Invalid 'min_keyword_length' in context: {min_keyword_length}. Must be a positive integer.")
            raise ValueError(f"'min_keyword_length' must be a positive integer, got {min_keyword_length}.")
        if not isinstance(top_n_keywords, int) or top_n_keywords <= 0:
            logger.error(f"Invalid 'top_n_keywords' in context: {top_n_keywords}. Must be a positive integer.")
            raise ValueError(f"'top_n_keywords' must be a positive integer, got {top_n_keywords}.")

        logger.debug(f"Extracting keywords with min_length={min_keyword_length}, top_n={top_n_keywords}")

        try:
            # 1. Normalize text: lowercase and remove non-alphanumeric characters
            text = data.lower()
            text = re.sub(r'[^a-z\s]', '', text) # Keep only lowercase letters and spaces

            # 2. Tokenize and filter words
            words = text.split()
            filtered_words = [word for word in words if len(word) >= min_keyword_length]

            if not filtered_words:
                logger.info("No words found matching minimum length criteria after processing.")
                return []

            # 3. Count word frequencies
            word_counts = Counter(filtered_words)

            # 4. Get the top N most common words
            # most_common returns a list of (word, count) tuples
            top_keywords_with_counts = word_counts.most_common(top_n_keywords)
            keywords = [word for word, count in top_keywords_with_counts]

            logger.info(f"Successfully extracted {len(keywords)} keywords.")
            logger.debug(f"Extracted keywords: {keywords}")

            return keywords

        except Exception as e:
            logger.exception(f"An unexpected error occurred during keyword extraction: {e}")
            # Re-raise or wrap the exception based on desired error propagation strategy
            raise RuntimeError(f"Failed to extract keywords due to an internal error: {e}") from e

