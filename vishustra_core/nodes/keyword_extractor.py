import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract keywords from a given text string.

    This node implements a simple, heuristic-based keyword extraction algorithm by:
    1. Converting the input text to lowercase.
    2. Stripping punctuation and non-alphanumeric characters (excluding spaces).
    3. Splitting the cleaned text into individual words.
    4. Filtering out words shorter than a specified minimum length and a predefined
       set of common stop words, along with any custom stop words provided in the context.
    5. Returning a unique list of the remaining words, optionally limited by a maximum count.

    Configuration via 'context' dictionary (optional):
    - 'min_word_length' (int): The minimum character length for a word to be
      considered a keyword. Defaults to 3.
    - 'max_keywords' (int): The maximum number of keywords to return. If 0, all
      unique keywords are returned. Defaults to 10.
    - 'custom_stop_words' (Set[str]): An additional set of words to filter out
      as stop words, complementing the default set.
    """

    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "for", "with", "by", "from", "to", "and", "or",
        "but", "not", "no", "yes", "this", "that", "these", "those", "it", "its",
        "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours",
        "you", "your", "yours", "they", "them", "their", "theirs", "i", "me", "my", "mine",
        "have", "has", "had", "do", "does", "did", "can", "could", "would", "should",
        "will", "may", "might", "must", "as", "if", "then", "than", "else", "when",
        "where", "why", "how", "what", "which", "who", "whom", "whose", "here", "there",
        "some", "any", "many", "much", "few", "more", "most", "all", "each", "every",
        "other", "another", "such", "so", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "about", "above", "below", "between",
        "through", "during", "before", "after", "while", "just", "now", "very", "too",
        "only", "own", "same", "s", "t", "don", "should", "d", "ll", "m", "o", "re", "ve", "y",
        "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
        "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords from a text string.

        Args:
            data (Any): The input data. Expected to be a string containing the text
                        from which keywords should be extracted.
            context (Dict[str, Any]): A dictionary providing runtime configuration
                                      parameters specific to this processing step.

        Returns:
            List[str]: A list of unique strings representing the extracted keywords.
                       The list is ordered based on the first appearance of the keyword.

        Raises:
            TypeError: If the input 'data' is not of type 'str'.
            ValueError: If 'data' is an empty string or consists only of whitespace
                        after stripping.
            Exception: For any other unexpected errors encountered during the
                       keyword extraction process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'.")
            raise TypeError(
                f"Input 'data' for {self.node_name} must be a string, "
                f"but got {type(data).__name__}.")

        text = data.strip()
        if not text:
            logger.warning(
                f"[{self.node_name}] Received an empty or whitespace-only string as input. "
                "Returning an empty list of keywords.")
            return []

        # Retrieve and validate context parameters
        min_word_length = context.get('min_word_length', 3)
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(
                f"[{self.node_name}] Invalid 'min_word_length' in context: {min_word_length}. "
                "Defaulting to 3.")
            min_word_length = 3

        max_keywords = context.get('max_keywords', 10)
        if not isinstance(max_keywords, int) or max_keywords < 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'max_keywords' in context: {max_keywords}. "
                "Defaulting to 10.")
            max_keywords = 10

        custom_stop_words = context.get('custom_stop_words', set())
        if not isinstance(custom_stop_words, Set):
            logger.warning(
                f"[{self.node_name}] Invalid 'custom_stop_words' in context. "
                f"Expected a Set, but got {type(custom_stop_words).__name__}. Ignoring and using only default.")
            custom_stop_words = set()

        all_stop_words = self._DEFAULT_STOP_WORDS.union(custom_stop_words)

        try:
            # Normalize text: convert to lowercase and remove non-alphabetic characters
            # keeping spaces to separate words. This is a simple form of tokenization.
            cleaned_text = re.sub(r'[^a-z\s]', '', text.lower())

            # Split text into words and filter based on length and stop words
            words = [
                word for word in cleaned_text.split()
                if len(word) >= min_word_length and word not in all_stop_words
            ]

            # Use dict.fromkeys to get unique words while preserving their original order
            unique_keywords = list(dict.fromkeys(words))

            # Apply max_keywords limit if specified
            if max_keywords > 0:
                extracted_keywords = unique_keywords[:max_keywords]
            else: # If max_keywords is 0, return all unique keywords
                extracted_keywords = unique_keywords

            logger.info(
                f"[{self.node_name}] Successfully extracted {len(extracted_keywords)} "
                f"keywords from input text (original length: {len(text)} characters)."
            )
            return extracted_keywords
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during keyword extraction: {e}",
                exc_info=True
            )
            raise Exception(f"Failed to process data in {self.node_name}: {e}") from e