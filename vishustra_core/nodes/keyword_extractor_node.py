import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract relevant keywords from an input text.

    This node tokenizes the input text, filters out words based on a configurable
    minimum length and a set of stop words, and returns a list of unique keywords.
    Case sensitivity for stop word matching can also be configured.
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
                  Non-string data will raise a TypeError.
            context: A dictionary containing operational context and configuration parameters:
                     - 'keyword_extractor_min_length': int, minimum length for a word to be
                       considered a keyword (default: 3). Must be non-negative.
                     - 'keyword_extractor_stopwords': Set[str] or List[str], a collection
                       of words to ignore during extraction (default: empty set).
                     - 'keyword_extractor_case_sensitive': bool, if True, stop word
                       matching is case-sensitive (default: False).

        Returns:
            A sorted list of unique extracted keywords (strings), preserving their
            original casing as found in the input text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If any context parameters are invalid (e.g., wrong type,
                        out of valid range).
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractorNode received non-string data of type '{type(data).__name__}'. "
                "Expected 'str' for keyword extraction."
            )
            raise TypeError("KeywordExtractorNode expects string input for data.")

        # --- Retrieve and validate configuration from context ---
        try:
            min_length = context.get('keyword_extractor_min_length', 3)
            if not isinstance(min_length, int) or min_length < 0:
                raise ValueError(
                    f"Invalid 'keyword_extractor_min_length' in context. "
                    f"Expected non-negative integer, got {min_length} ({type(min_length).__name__})."
                )

            stopwords_raw = context.get('keyword_extractor_stopwords', set())
            if not isinstance(stopwords_raw, (set, list, tuple)):
                raise ValueError(
                    f"Invalid 'keyword_extractor_stopwords' in context. "
                    f"Expected a set, list, or tuple of strings, got {type(stopwords_raw).__name__}."
                )
            
            case_sensitive = context.get('keyword_extractor_case_sensitive', False)
            if not isinstance(case_sensitive, bool):
                raise ValueError(
                    f"Invalid 'keyword_extractor_case_sensitive' in context. "
                    f"Expected boolean, got {case_sensitive} ({type(case_sensitive).__name__})."
                )

            # Prepare stopwords based on case sensitivity
            if case_sensitive:
                stopwords: Set[str] = {str(word) for word in stopwords_raw}
            else:
                stopwords: Set[str] = {str(word).lower() for word in stopwords_raw}

        except ValueError as e:
            logger.error(f"KeywordExtractorNode configuration error: {e}")
            raise

        # --- Tokenization and Filtering ---
        extracted_keywords_set: Set[str] = set()

        # Use regex to find sequences of word characters. \b ensures word boundaries.
        # \w includes alphanumeric and underscore, which is suitable for many keyword definitions.
        tokens = re.findall(r'\b\w+\b', data)

        for original_token in tokens:
            # Clean token by keeping only alphanumeric characters.
            # This handles cases like "word." becoming "word".
            cleaned_token = ''.join(filter(str.isalnum, original_token))

            if not cleaned_token: # Skip empty strings resulting from cleaning
                continue

            # Determine the form of the word for comparison (e.g., lowercased or original)
            word_for_comparison = cleaned_token.lower() if not case_sensitive else cleaned_token

            # Check minimum length
            if len(cleaned_token) < min_length:
                continue

            # Check against stopwords set
            is_stop_word = word_for_comparison in stopwords

            if not is_stop_word:
                # Add the keyword in its cleaned, original casing to the result set
                extracted_keywords_set.add(cleaned_token)

        logger.info(
            f"KeywordExtractorNode successfully processed data. "
            f"Extracted {len(extracted_keywords_set)} unique keywords."
        )
        # Return a sorted list for consistent and deterministic output
        return sorted(list(extracted_keywords_set))