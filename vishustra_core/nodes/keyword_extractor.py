import logging
import re
from typing import Any, Dict, List, Set, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text.
    This implementation simulates keyword extraction by tokenizing the input,
    filtering common words (stopwords), and short tokens, returning a unique
    set of significant words.
    """

    def __init__(
        self,
        stopwords: Union[List[str], Set[str], None] = None,
        min_keyword_length: int = 3,
        max_keywords: int = 10
    ):
        """
        Initializes the KeywordExtractorNode with optional configuration.

        Args:
            stopwords: An optional list or set of words to exclude from keywords.
                       If None, a default set of common English stopwords is used.
            min_keyword_length: The minimum character length for a token to be
                                considered a keyword.
            max_keywords: The maximum number of unique keywords to return.
        """
        if stopwords is None:
            # A common set of English stopwords for demonstration purposes.
            # In a production system, this would likely be loaded from a resource
            # or configurable via a robust mechanism.
            self._stopwords = {
                "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                "and", "or", "but", "for", "nor", "to", "of", "in", "on", "at",
                "by", "with", "from", "as", "he", "she", "it", "they", "we", "you",
                "i", "me", "him", "her", "us", "them", "my", "your", "his", "hers",
                "its", "our", "their", "this", "that", "these", "those", "can",
                "will", "would", "should", "could", "may", "might", "must", "do",
                "does", "did", "have", "has", "had", "not", "no", "yes", "up",
                "down", "out", "about", "above", "below", "through", "before",
                "after", "again", "further", "then", "once", "here", "there",
                "when", "where", "why", "how", "all", "any", "both", "each",
                "few", "more", "most", "other", "some", "such", "nor",
                "only", "own", "same", "so", "than", "too", "very", "s", "t",
                "just", "don", "should", "now", "ve", "ll", "re", "m", "o", "d"
            }
        else:
            self._stopwords = set(stopwords)

        if not isinstance(min_keyword_length, int) or min_keyword_length < 1:
            logger.warning(
                "Invalid min_keyword_length provided. Defaulting to 3. "
                "Received: %s (type: %s)", min_keyword_length, type(min_keyword_length).__name__
            )
            self._min_keyword_length = 3
        else:
            self._min_keyword_length = min_keyword_length

        if not isinstance(max_keywords, int) or max_keywords < 1:
            logger.warning(
                "Invalid max_keywords provided. Defaulting to 10. "
                "Received: %s (type: %s)", max_keywords, type(max_keywords).__name__
            )
            self._max_keywords = 10
        else:
            self._max_keywords = max_keywords

        logger.info(
            "KeywordExtractorNode initialized with min_keyword_length=%d, "
            "max_keywords=%d, and %d stopwords.",
            self._min_keyword_length, self._max_keywords, len(self._stopwords)
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract keywords.

        Args:
            data: The input text data from which to extract keywords.
            context: A dictionary containing context-specific information for the
                     current processing pipeline run. This node does not currently
                     utilize the context for its primary logic.

        Returns:
            A list of extracted keywords (strings), sorted alphabetically.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If an unexpected error occurs during keyword extraction.
        """
        if not isinstance(data, str):
            logger.error(
                "Invalid input data type for KeywordExtractorNode. "
                "Expected str, but received %s.", type(data).__name__
            )
            raise TypeError(
                f"KeywordExtractorNode expects string data, "
                f"but received {type(data).__name__}"
            )

        extracted_keywords: List[str] = []
        try:
            # Simple tokenization: convert to lowercase and split by non-alphanumeric
            # A real-world implementation would typically use a more sophisticated
            # NLP tokenizer (e.g., from NLTK, spaCy) for better accuracy with
            # contractions, hyphenated words, etc.
            text_lower = data.lower()
            text_tokens = re.findall(r'\b[a-z]+\b', text_lower)

            # Filter tokens based on length and stopwords
            unique_keywords: Set[str] = set()
            for token in text_tokens:
                if (
                    token and
                    len(token) >= self._min_keyword_length and
                    token not in self._stopwords
                ):
                    unique_keywords.add(token)

            # Convert set to list, sort alphabetically, and limit the number of keywords
            # For a real system, keywords might be ranked by frequency, TF-IDF,
            # or other relevance metrics before selection.
            extracted_keywords = sorted(list(unique_keywords))[:self._max_keywords]

            if not extracted_keywords:
                logger.info(
                    "No keywords extracted from data of length %d after filtering.",
                    len(data)
                )
            else:
                logger.debug(
                    "Successfully extracted %d keywords: %s",
                    len(extracted_keywords), extracted_keywords
                )

        except Exception as e:
            logger.exception(
                "An unexpected error occurred during keyword extraction: %s", e
            )
            raise ValueError(f"Keyword extraction failed: {e}") from e

        return extracted_keywords