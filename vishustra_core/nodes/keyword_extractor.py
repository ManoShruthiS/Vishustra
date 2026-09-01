import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node that extracts keywords from a given text string.

    This node performs a simplified keyword extraction by tokenizing the input
    text, converting it to lowercase, removing common stop words, filtering
    out short or non-alphabetic tokens, and then returning a list of unique
    significant words.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data.

        Expected `data` type: `str`.
        Expected `context` keys:
        - `stop_words` (Optional[List[str]]): A list of words to ignore. Defaults to an empty set.
        - `min_word_length` (Optional[int]): Minimum character length for a word to be considered a keyword. Defaults to 3.
        - `max_keywords` (Optional[int]): The maximum number of keywords to return. Defaults to `None` (no limit).

        Args:
            data: The input text as a string from which to extract keywords.
            context: A dictionary containing operational parameters like stop words,
                     minimum word length, and maximum keywords.

        Returns:
            A list of unique keywords extracted from the text, sorted alphabetically.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `min_word_length` or `max_keywords` in context are non-positive.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for KeywordExtractorNode. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(
                f"KeywordExtractorNode requires 'data' to be a string for processing. "
                f"Received type: {type(data).__name__}"
            )

        text = data.lower()
        
        # Configure from context with sensible defaults
        stop_words: Set[str] = set(context.get('stop_words', []))
        min_word_length: int = context.get('min_word_length', 3)
        max_keywords: int | None = context.get('max_keywords')

        if not isinstance(min_word_length, int) or min_word_length <= 0:
            logger.error(f"Invalid 'min_word_length' in context: {min_word_length}. Must be a positive integer.")
            raise ValueError("Context 'min_word_length' must be a positive integer.")
        if max_keywords is not None and (not isinstance(max_keywords, int) or max_keywords <= 0):
            logger.error(f"Invalid 'max_keywords' in context: {max_keywords}. Must be a positive integer or None.")
            raise ValueError("Context 'max_keywords' must be a positive integer or None.")

        # Simple tokenization: split by non-alphanumeric characters and filter
        words = re.findall(r'\b[a-z]+\b', text)
        
        extracted_keywords: Set[str] = set()
        for word in words:
            if word not in stop_words and len(word) >= min_word_length:
                extracted_keywords.add(word)
        
        # Sort for deterministic output and potential slicing
        sorted_keywords = sorted(list(extracted_keywords))

        if max_keywords is not None:
            sorted_keywords = sorted_keywords[:max_keywords]

        logger.debug(f"KeywordExtractorNode processed data and extracted {len(sorted_keywords)} keywords.")
        return sorted_keywords

# Example of basic usage (not part of the node itself, but for testing/understanding)
if __name__ == '__main__':
    # Setup basic logging for demonstration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    extractor = KeywordExtractorNode()
    
    # Test case 1: Basic extraction
    text1 = "The quick brown fox jumps over the lazy dog. Fox is quick."
    context1 = {
        'stop_words': ['the', 'is', 'over', 'a', 'an'],
        'min_word_length': 3
    }
    keywords1 = extractor.process(text1, context1)
    logger.info(f"Text 1 Keywords: {keywords1}")
    assert sorted(keywords1) == ['brown', 'dog', 'fox', 'jumps', 'lazy', 'quick']

    # Test case 2: Different stop words and max keywords
    text2 = "Learning Python is fun and Python is powerful. We love Python."
    context2 = {
        'stop_words': ['is', 'and', 'we', 'love'],
        'min_word_length': 4,
        'max_keywords': 2
    }
    keywords2 = extractor.process(text2, context2)
    logger.info(f"Text 2 Keywords (max 2): {keywords2}")
    assert sorted(keywords2) == ['fun', 'learning']

    # Test case 3: Empty text
    text3 = ""
    keywords3 = extractor.process(text3, {})
    logger.info(f"Text 3 Keywords (empty): {keywords3}")
    assert keywords3 == []

    # Test case 4: No significant words
    text4 = "a is the of and"
    keywords4 = extractor.process(text4, {'min_word_length': 2, 'stop_words': ['a', 'is', 'the', 'of', 'and']})
    logger.info(f"Text 4 Keywords (no significant): {keywords4}")
    assert keywords4 == []

    # Test case 5: Error handling - wrong data type
    try:
        extractor.process(123, {})
    except TypeError as e:
        logger.info(f"Caught expected error: {e}")
        assert "requires 'data' to be a string" in str(e)

    # Test case 6: Error handling - invalid min_word_length
    try:
        extractor.process("some text", {'min_word_length': 0})
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")
        assert "must be a positive integer" in str(e)

    # Test case 7: Error handling - invalid max_keywords
    try:
        extractor.process("some text", {'max_keywords': -1})
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")
        assert "must be a positive integer or None" in str(e)
        
    logger.info("All KeywordExtractorNode tests passed.")