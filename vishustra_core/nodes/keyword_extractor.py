import logging
import re
from typing import Any, Dict, List, Set, Tuple

# Assuming BaseNode is located here based on the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract relevant keywords from textual data.

    This node takes a string as input and applies a set of configurable rules
    to identify and return a list of significant keywords. The current implementation
    employs heuristics like word length, exclusion of common words, and frequency
    analysis to simulate a robust extraction process.
    """

    # A predefined set of common English stopwords for basic filtering
    _common_english_words: Set[str] = {
        "the", "and", "a", "an", "is", "it", "in", "on", "of", "to", "for", "with",
        "as", "at", "by", "from", "he", "she", "i", "we", "you", "they", "that",
        "this", "but", "or", "not", "be", "have", "do", "say", "get", "make", "go",
        "know", "take", "see", "come", "think", "look", "want", "give", "use",
        "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call",
        "good", "new", "first", "last", "long", "great", "little", "own", "other",
        "old", "right", "big", "high", "different", "small", "large", "next",
        "early", "important", "few", "public", "bad", "same", "able", "would",
        "could", "should", "will", "can", "may", "much", "many", "such", "also",
        "about", "all", "any", "both", "each", "every", "here", "how", "if", "into",
        "like", "most", "no", "nor", "only", "other", "our", "out", "over", "some",
        "than", "then", "there", "these", "through", "under", "up", "very", "was",
        "when", "where", "which", "while", "who", "whom", "why", "your"
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract key terms or phrases.

        This method expects the 'data' parameter to be a string (text) and uses
        configuration from the 'context' dictionary to guide the extraction process.
        It tokenizes the text, filters words based on length and commonality,
        and then selects the most frequent candidate keywords up to a specified limit.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        from which keywords should be extracted.
            context (Dict[str, Any]): A dictionary providing configuration for the
                                       keyword extraction logic. Expected keys:
                                       - 'min_word_length' (int, optional): The minimum
                                         length a word must have to be considered a keyword.
                                         Defaults to 4.
                                       - 'max_keywords' (int, optional): The maximum number
                                         of keywords to return. Defaults to 10.
                                       - 'exclude_common_words' (bool, optional): If True,
                                         common English stopwords are filtered out. Defaults to True.
                                       - 'pattern' (str, optional): A regex pattern to split
                                         the text into words. Defaults to r'\b\w+\b'.

        Returns:
            List[str]: A list of strings, each representing an extracted keyword.
                       The list is sorted by frequency (descending) and then
                       alphabetically (ascending) for consistency. Returns an
                       empty list if the input is empty, invalid, or no keywords
                       are found.

        Raises:
            ValueError: If 'data' is not a string, or if context parameters are
                        of an invalid type or value.
            Exception: Catches and logs any unexpected errors during the processing
                       logic, re-raising them to indicate a failure.
        """
        # --- Input Data Validation ---
        if not isinstance(data, str):
            logger.error(f"KeywordExtractor received invalid data type. Expected 'str', got {type(data)}.")
            raise ValueError(
                f"Invalid input data type for KeywordExtractor. Expected a string, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.info("Empty or whitespace-only input string provided to KeywordExtractor. Returning an empty list.")
            return []

        # --- Context Parameter Processing ---
        try:
            min_word_length = int(context.get('min_word_length', 4))
            max_keywords = int(context.get('max_keywords', 10))
            exclude_common_words = bool(context.get('exclude_common_words', True))
            word_split_pattern = str(context.get('pattern', r'\b\w+\b'))

            if min_word_length <= 0:
                logger.warning(
                    f"Invalid 'min_word_length' ({min_word_length}) provided. "
                    f"Clamping to default value: 4."
                )
                min_word_length = 4
            if max_keywords <= 0:
                logger.warning(
                    f"Invalid 'max_keywords' ({max_keywords}) provided. "
                    f"Clamping to default value: 10."
                )
                max_keywords = 10

        except (TypeError, ValueError) as e:
            logger.error(
                f"Failed to parse context parameters for KeywordExtractor: {e}. "
                f"Context: {context}"
            )
            raise ValueError(f"Invalid type or value for context parameter: {e}") from e

        logger.debug(
            f"KeywordExtractor initialized with: min_word_length={min_word_length}, "
            f"max_keywords={max_keywords}, exclude_common_words={exclude_common_words}, "
            f"pattern='{word_split_pattern}'"
        )

        # --- Keyword Extraction Logic ---
        extracted_keywords: List[str] = []
        try:
            # Tokenize text using the specified pattern
            words = re.findall(word_split_pattern, data.lower())

            candidate_keywords: List[str] = []
            for word in words:
                # Basic cleanup: remove leading/trailing non-alphanumeric (if pattern allows)
                cleaned_word = word.strip(".,!?;:\"'()[]{}/\\-").lower()

                if len(cleaned_word) >= min_word_length:
                    if exclude_common_words and cleaned_word in self._common_english_words:
                        continue
                    if cleaned_word: # Ensure not empty after cleaning
                        candidate_keywords.append(cleaned_word)

            # Count word frequencies
            word_counts: Dict[str, int] = {}
            for word in candidate_keywords:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Sort keywords by frequency (descending) and then alphabetically (ascending)
            # for stable results in case of ties.
            sorted_keywords: List[Tuple[str, int]] = sorted(
                word_counts.items(),
                key=lambda item: (-item[1], item[0])
            )

            # Select the top N keywords
            extracted_keywords = [word for word, count in sorted_keywords[:max_keywords]]

            logger.info(f"Successfully extracted {len(extracted_keywords)} keywords.")

        except Exception as e:
            logger.exception(f"An unexpected error occurred during keyword extraction processing: {e}")
            raise # Re-raise the exception after logging

        return extracted_keywords