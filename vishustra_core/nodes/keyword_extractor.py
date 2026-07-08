import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract simulated keywords from text data.
    This node identifies potential keywords based on configurable criteria such as
    minimum word length and a customizable list of stop words.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data, expected to be a string, to identify and
        extract simulated keywords.

        The behavior of the extraction can be configured through the `context` dictionary:
        - 'min_word_length': An integer specifying the minimum character length
          a word must have to be considered a keyword (default: 3).
        - 'stop_words': A collection (e.g., list, tuple, set) of words to be
          ignored during the extraction process. These are case-insensitive.
          (default: a predefined set of common English stop words).

        Args:
            data: The input text from which keywords are to be extracted.
                  Expected type is `str`.
            context: A dictionary potentially containing configuration parameters
                     for the keyword extraction.

        Returns:
            A sorted list of unique strings representing the extracted keywords.

        Raises:
            ValueError: If the `data` input is not a string.
            Exception: Propagates any unexpected errors encountered during the
                       processing, after logging the incident.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', but received '%s'.",
                self.node_name,
                type(data).__name__,
            )
            raise ValueError(
                f"Input data for '{self.node_name}' must be a string, "
                f"but received '{type(data).__name__}'."
            )

        try:
            # Retrieve configuration from context, applying defaults if not present.
            min_word_length = context.get("min_word_length", 3)
            if not isinstance(min_word_length, int) or min_word_length < 0:
                logger.warning(
                    "[%s] 'min_word_length' in context is invalid (%s). Using default: 3.",
                    self.node_name, min_word_length
                )
                min_word_length = 3

            # Default English stop words
            default_stop_words = {
                "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
                "in", "on", "at", "for", "with", "as", "by", "of", "to", "from",
                "it", "its", "i", "you", "he", "she", "we", "they", "them", "us",
                "this", "that", "these", "those", "can", "will", "would", "could",
                "have", "has", "had", "do", "does", "did", "be", "been", "being",
                "not", "no", "yes", "me", "my", "your", "his", "her", "our", "their",
                "s", "t", "d", "ll", "m", "re", "ve", "y", "am" # common contractions and short words
            }

            custom_stop_words_raw = context.get("stop_words", [])
            if not isinstance(custom_stop_words_raw, (list, tuple, set)):
                logger.warning(
                    "[%s] 'stop_words' in context is not a list, tuple, or set (%s). Ignoring custom stop words.",
                    self.node_name, type(custom_stop_words_raw).__name__
                )
                custom_stop_words_set = set()
            else:
                custom_stop_words_set = {str(word).lower() for word in custom_stop_words_raw}

            # Combine default and custom stop words for efficient lookup
            all_stop_words: Set[str] = default_stop_words.union(custom_stop_words_set)

            # Convert input text to lowercase to ensure case-insensitive processing.
            text_lower = data.lower()
            # Use regex to find all word characters, effectively splitting by non-word chars
            # and handling most punctuation.
            words = re.findall(r'\b\w+\b', text_lower)

            extracted_keywords = set()
            for word in words:
                if len(word) >= min_word_length and word not in all_stop_words:
                    extracted_keywords.add(word)

            # Sort the unique keywords for deterministic output.
            result = sorted(list(extracted_keywords))
            logger.debug(
                "[%s] Successfully extracted %d keywords from the input data.",
                self.node_name,
                len(result),
            )
            return result
        except Exception as e:
            logger.exception(
                "[%s] An unforeseen error occurred during the keyword extraction process.",
                self.node_name,
            )
            # Re-raise the exception to allow upstream error handling.
            raise
