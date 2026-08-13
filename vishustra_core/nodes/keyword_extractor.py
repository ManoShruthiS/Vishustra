import logging
import re
from typing import Any, Dict, List, Set, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class KeywordExtractor(BaseNode):
    """
    A Vishustra node designed to extract keywords from textual data.

    This node performs a series of steps including text cleaning, tokenization,
    stop-word removal, and length-based filtering to identify relevant keywords
    within the input text. It can be configured via the context dictionary
    to use custom stop words or adjust the minimum keyword length.
    """

    # Default set of common English stop words.
    DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now", "has", "have", "had", "do", "does", "did", "him",
        "his", "her", "hers", "me", "my", "mine", "you", "your", "yours",
        "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs"
    }
    # Default minimum length for an extracted keyword.
    DEFAULT_MIN_KEYWORD_LENGTH: int = 3

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        The `context` dictionary can be used to customize the keyword extraction:
        - `'stop_words'`: A `list` or `set` of strings representing custom stop words.
                          If provided, these will override the default stop words.
        - `'min_keyword_length'`: An `int` specifying the minimum character length
                                  for a word to be considered a keyword.
                                  Defaults to `DEFAULT_MIN_KEYWORD_LENGTH`.

        Args:
            data: The input text as a string from which keywords are to be extracted.
            context: A dictionary potentially containing configuration parameters
                     for the extraction process.

        Returns:
            A list of unique strings, each representing an extracted keyword,
            sorted alphabetically.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'min_keyword_length' in `context` is not a positive integer.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for '{self.node_name}'. "
                f"Expected str, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.lower()
        logger.debug(f"Initiating keyword extraction for text of length: {len(text)}.")

        # Determine stop words to use
        stop_words_from_context = context.get('stop_words')
        stop_words: Set[str] = self.DEFAULT_STOP_WORDS
        if stop_words_from_context is not None:
            if isinstance(stop_words_from_context, (list, set)):
                stop_words = set(stop_words_from_context)
                logger.debug(f"Using custom stop words from context (count: {len(stop_words)}).")
            else:
                logger.warning(
                    f"Invalid type for 'stop_words' in context for '{self.node_name}'. "
                    f"Expected list or set, but got {type(stop_words_from_context).__name__}. "
                    "Falling back to default stop words."
                )

        # Determine minimum keyword length
        min_keyword_length: int = self.DEFAULT_MIN_KEYWORD_LENGTH
        min_length_from_context = context.get('min_keyword_length')
        if min_length_from_context is not None:
            if isinstance(min_length_from_context, int) and min_length_from_context > 0:
                min_keyword_length = min_length_from_context
                logger.debug(f"Using custom minimum keyword length: {min_keyword_length}.")
            else:
                error_msg = (
                    f"Invalid 'min_keyword_length' in context for '{self.node_name}'. "
                    f"Expected a positive integer, but got {type(min_length_from_context).__name__} "
                    f"with value '{min_length_from_context}'. Please provide a valid positive integer."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Basic text cleaning: remove non-alphanumeric characters (except spaces)
        # and tokenize into words.
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        words = cleaned_text.split()

        extracted_keywords: Set[str] = set()  # Use a set to automatically handle duplicates
        for word in words:
            # Filter words based on stop-word list and minimum length
            if word not in stop_words and len(word) >= min_keyword_length:
                extracted_keywords.add(word)

        # Convert the set to a list and sort for consistent output
        final_keywords = sorted(list(extracted_keywords))
        logger.info(f"Successfully extracted {len(final_keywords)} unique keywords.")
        logger.debug(f"Extracted keywords: {final_keywords}")

        return final_keywords