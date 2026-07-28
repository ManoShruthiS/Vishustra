import logging
import re
from typing import Any, Dict, List, Set

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra node designed to extract keywords from a given text string.

    This node performs a basic keyword extraction by tokenizing the input
    text, converting it to lowercase, removing punctuation, and filtering out
    common stop words. It then returns a list of unique words that remain.

    The extraction process can be refined using context parameters, allowing
    control over factors like minimum word length and the maximum number of
    keywords to be returned.
    """

    # A predefined set of common English stop words for filtering
    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
        "for", "with", "at", "by", "on", "in", "of", "to", "from", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
        "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
        "shouldn", "wasn", "weren", "won", "wouldn"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords from text.

        The method expects a string as input data and returns a list of
        extracted keywords. Context parameters allow for customisation of
        the extraction logic.

        Args:
            data (Any): The input data. Expected to be a string containing
                        the text from which to extract keywords.
            context (Dict[str, Any]): A dictionary providing execution context
                                      and configuration parameters:
                                      - 'min_word_length' (int, default: 3):
                                        The minimum character length for a word
                                        to be considered a keyword. Words shorter
                                        than this will be filtered out.
                                      - 'max_keywords' (int, default: None):
                                        If provided and positive, limits the
                                        number of keywords returned to this value.
                                        If None, all filtered keywords are returned.
                                      - 'custom_stop_words' (List[str], default: []):
                                        A list of additional words to treat as
                                        stop words and filter out during extraction.

        Returns:
            List[str]: A list of unique, extracted keywords, potentially
                       limited by `max_keywords` and filtered by length
                       and stop words.

        Raises:
            ValueError: If the input 'data' is not a string, as this node
                        is specifically designed for text processing.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractor received invalid input data type. Expected 'str', "
                f"but got '{type(data).__name__}'. Please provide a string."
            )
            raise ValueError("KeywordExtractor expects string input for data.")

        # Convert text to lowercase to ensure case-insensitive matching
        text = data.lower()
        
        # Use regex to find all sequences of word characters, effectively
        # tokenizing and removing punctuation.
        words = re.findall(r'\b\w+\b', text)

        # Retrieve context parameters with sensible defaults
        min_word_length = context.get('min_word_length', 3)
        max_keywords = context.get('max_keywords')
        custom_stop_words_list = context.get('custom_stop_words', [])

        # Validate min_word_length
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(
                f"Invalid 'min_word_length' in context: {min_word_length}. "
                f"Defaulting to 3. Expected a positive integer."
            )
            min_word_length = 3

        # Combine default stop words with any custom ones provided in context
        all_stop_words = self._DEFAULT_STOP_WORDS.union(set(custom_stop_words_list))

        filtered_keywords = []
        for word in words:
            # Filter based on stop words and minimum word length
            if word not in all_stop_words and len(word) >= min_word_length:
                filtered_keywords.append(word)

        # Ensure uniqueness while preserving the order of first appearance
        # dict.fromkeys() is a common and efficient way to do this for Python 3.7+
        unique_keywords = list(dict.fromkeys(filtered_keywords))

        final_keywords: List[str] = unique_keywords

        # Apply the maximum number of keywords limit if specified and valid
        if max_keywords is not None:
            if isinstance(max_keywords, int) and max_keywords > 0:
                final_keywords = unique_keywords[:max_keywords]
                logger.debug(
                    f"KeywordExtractor limited the output to {len(final_keywords)} "
                    f"keywords as per 'max_keywords' setting ({max_keywords})."
                )
            else:
                logger.warning(
                    f"Invalid 'max_keywords' in context: {max_keywords}. "
                    f"Expected a positive integer or None. Ignoring limit."
                )

        logger.info(
            f"KeywordExtractor successfully processed text, extracting "
            f"{len(final_keywords)} keywords."
        )
        logger.debug(f"Extracted keywords: {final_keywords}")

        return final_keywords