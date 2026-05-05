import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A data processing node for Vishustra that extracts keywords from text.

    This node performs a tokenization, cleansing, and frequency-based selection
    to identify relevant keywords within an input text string. It's configurable
    with parameters for stop words, minimum keyword length, and the maximum
    number of keywords to return.
    """

    # A simple, illustrative set of common English stop words.
    # In a real-world scenario, this would likely be loaded from a more comprehensive
    # external resource or configurable via constructor.
    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "for", "with", "on", "at", "by", "from",
        "up", "down", "in", "out", "over", "under", "about", "above", "below",
        "through", "between", "among", "do", "does", "did", "not", "no", "yes",
        "he", "she", "it", "we", "they", "you", "i", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those",
        "can", "could", "will", "would", "should", "may", "might", "must", "have", "has", "had",
        "but", "if", "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom",
        "than", "as", "such", "only", "also", "just", "very", "too", "so", "much", "more", "most",
        "few", "less", "many", "some", "any", "all", "each", "every", "both", "either", "neither",
        "while", "before", "after", "since", "until", "though", "although", "unless", "except",
        "like", "into", "onto", "off", "back", "again", "then", "there", "here", "away", "always",
        "never", "often", "seldom", "sometimes", "usually", "ever", "once", "twice", "said", "say",
        "get", "got", "go", "went", "going", "make", "made", "making", "take", "took", "taking",
        "come", "came", "coming", "see", "saw", "seeing", "find", "found", "finding", "give",
        "gave", "giving", "tell", "told", "telling", "work", "worked", "working", "wouldn", "couldn"
    }

    def __init__(
        self,
        min_keyword_length: int = 3,
        max_keywords: int = 10,
        custom_stop_words: Optional[Set[str]] = None
    ):
        """
        Initializes the KeywordExtractorNode with default or custom configuration.

        Args:
            min_keyword_length: The minimum character length for a word to be considered a keyword.
            max_keywords: The maximum number of top keywords to return.
            custom_stop_words: An optional set of additional stop words to filter out.
                               These will augment the node's default stop word list.
        """
        super().__init__()
        self._min_keyword_length = min_keyword_length
        self._max_keywords = max_keywords
        self._stop_words = self._DEFAULT_STOP_WORDS.copy()
        if custom_stop_words:
            self._stop_words.update(custom_stop_words)
        logger.debug(
            f"KeywordExtractorNode initialized with min_keyword_length={self._min_keyword_length}, "
            f"max_keywords={self._max_keywords}, stop_words_count={len(self._stop_words)}"
        )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts a list of keywords from the input text data.

        The method processes the input text by tokenizing it, removing punctuation,
        filtering out stop words and short words, and then selecting the most frequent
        words as keywords. Configuration can be overridden via the `context`.

        Args:
            data: The input text as a string from which to extract keywords.
            context: A dictionary containing additional context or configuration.
                     This can include 'min_keyword_length', 'max_keywords',
                     and 'custom_stop_words' to override node's initialization settings.

        Returns:
            A list of strings, where each string is an extracted keyword.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the processed text yields no extractable keywords.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type: Expected str, got {type(data).__name__}")
            raise TypeError(f"KeywordExtractorNode requires string input, but received {type(data).__name__}")

        if not data.strip():
            logger.info("Received empty or whitespace-only string, returning an empty list of keywords.")
            return []

        # Override configuration from context if available
        current_min_keyword_length = context.get('min_keyword_length', self._min_keyword_length)
        current_max_keywords = context.get('max_keywords', self._max_keywords)
        current_stop_words = self._stop_words.copy()
        context_custom_stop_words = context.get('custom_stop_words')
        if isinstance(context_custom_stop_words, (set, list)):
            current_stop_words.update(set(word.lower() for word in context_custom_stop_words))
        elif context_custom_stop_words is not None:
            logger.warning(
                f"Invalid type for 'custom_stop_words' in context: {type(context_custom_stop_words).__name__}. "
                "Expected set or list. Ignoring."
            )

        logger.debug(
            f"Processing text for keywords using min_length={current_min_keyword_length}, "
            f"max_keywords={current_max_keywords}, text_length={len(data)}"
        )

        # 1. Normalize and tokenize text
        text_lower = data.lower()
        # Remove punctuation but keep spaces to split words correctly
        cleaned_text = re.sub(r'[^\w\s]', ' ', text_lower)
        words = cleaned_text.split()

        # 2. Filter words based on stop words and length
        filtered_words = [
            word for word in words
            if word not in current_stop_words and len(word) >= current_min_keyword_length
        ]

        if not filtered_words:
            logger.warning(f"No significant words found after filtering for text: '{data[:50]}...'")
            return []

        # 3. Count word frequencies
        word_counts = Counter(filtered_words)

        # 4. Extract top keywords
        # Counter.most_common returns a list of (word, count) tuples
        top_keyword_tuples = word_counts.most_common(current_max_keywords)
        extracted_keywords = [keyword for keyword, _count in top_keyword_tuples]

        if not extracted_keywords:
            logger.warning(f"Could not extract any keywords from input text after frequency analysis: '{data[:50]}...'")
            raise ValueError("No extractable keywords found after processing.")

        logger.info(f"Successfully extracted {len(extracted_keywords)} keywords.")
        logger.debug(f"Extracted keywords: {extracted_keywords}")
        return extracted_keywords

# Example of how to integrate with a Vishustra orchestration (conceptual):
# if __name__ == '__main__':
#     # Configure logging for standalone testing
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     extractor = KeywordExtractorNode(min_keyword_length=4, max_keywords=5)
#
#     test_text_1 = "The quick brown fox jumps over the lazy dog. The fox is very quick."
#     test_text_2 = "Artificial intelligence research explores machine learning models."
#     test_text_3 = "Python is a powerful programming language widely used for backend development."
#     test_text_4 = "         " # Empty string test
#     test_text_5 = "a an the is are" # Only stop words
#     test_text_6 = "short words only like cat dog run"
#
#     print(f"Node Name: {extractor.node_name}")
#
#     try:
#         print(f"Text 1 Keywords: {extractor.process(test_text_1, {})}")
#         print(f"Text 2 Keywords: {extractor.process(test_text_2, {'max_keywords': 3, 'min_keyword_length': 6})}")
#         print(f"Text 3 Keywords: {extractor.process(test_text_3, {'custom_stop_words': ['python']})}")
#         print(f"Text 4 Keywords: {extractor.process(test_text_4, {})}")
#         print(f"Text 5 Keywords: {extractor.process(test_text_5, {})}")
#         print(f"Text 6 Keywords: {extractor.process(test_text_6, {})}")
#         # Invalid input type
#         # print(f"Invalid input: {extractor.process(123, {})}")
#     except Exception as e:
#         print(f"Error during processing: {e}")