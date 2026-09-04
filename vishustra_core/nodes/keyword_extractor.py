import logging
import string
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra node that extracts potential keywords from input text.

    This node takes text (either directly as a string or within a dictionary under the 'text' key)
    and processes it to identify a list of significant words.
    The current implementation performs basic tokenization, lowercasing, stop word removal,
    and filters out short words to simulate keyword extraction.
    """

    _DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "else", "for", "with", "at", "from",
        "in", "on", "by", "to", "of", "do", "does", "did", "not", "no", "yes",
        "it", "its", "he", "she", "we", "they", "you", "your", "my", "our",
        "i", "me", "us", "him", "her", "them", "this", "that", "these", "those",
        "which", "what", "where", "when", "why", "how", "all", "any", "some",
        "such", "only", "own", "so", "than", "too", "very", "s", "t", "can",
        "will", "just", "don", "should", "now", "would", "could", "get", "has",
        "have", "had", "say", "said", "go", "went", "going", "make", "made",
        "making", "see", "saw", "seeing", "take", "took", "taking", "know",
        "knew", "knowing", "think", "thought", "thinking", "come", "came",
        "coming", "want", "wanted", "wanting", "look", "looked", "looking",
        "give", "gave", "giving", "use", "used", "using", "find", "found",
        "finding", "tell", "told", "telling", "ask", "asked", "asking",
        "work", "worked", "working", "may", "might", "must", "need", "should",
        "like", "good", "bad", "new", "old", "first", "last", "long", "short",
        "high", "low", "big", "small", "many", "much", "more", "most", "other",
        "next", "every", "few", "little", "right", "left", "up", "down", "out",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "what", "who", "whom", "whose", "would",
        "still", "before", "after", "through", "while", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "will", "just", "don", "should", "now"
    }
    _MIN_WORD_LENGTH: int = 3

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input data.

        The input `data` is expected to be either:
        1. A string containing the text to be processed.
        2. A dictionary with a 'text' key, whose value is the string to be processed.

        The `context` dictionary can optionally contain:
        - 'stop_words': A set of strings to be excluded from keywords.
        - 'min_word_length': An integer representing the minimum length for a word to be considered a keyword.

        Args:
            data: The input text data (string or dict with 'text' key).
            context: A dictionary for node-specific configuration.

        Returns:
            A list of unique strings identified as keywords, sorted alphabetically.
            Returns an empty list if no valid text is provided or no keywords are found.

        Raises:
            ValueError: If the input `data` is not a string or a dictionary
                        containing a 'text' key with a string value.
        """
        text_to_process: str = ""
        # Allow context to override default parameters
        stop_words: Set[str] = context.get('stop_words', self._DEFAULT_STOP_WORDS)
        min_word_length: int = context.get('min_word_length', self._MIN_WORD_LENGTH)

        if isinstance(data, str):
            text_to_process = data
            logger.debug(f"KeywordExtractor: Processing text directly from string input (length: {len(data)}).")
        elif isinstance(data, dict):
            if 'text' in data and isinstance(data['text'], str):
                text_to_process = data['text']
                logger.debug(f"KeywordExtractor: Processing text from 'text' key in dictionary input (length: {len(text_to_process)}).")
            else:
                logger.error(
                    f"KeywordExtractor: Dictionary input must contain a 'text' key with a string value. "
                    f"Received keys: {data.keys() if isinstance(data, dict) else 'N/A'} and value type for 'text': "
                    f"{type(data.get('text')) if isinstance(data, dict) and 'text' in data else 'N/A'}"
                )
                raise ValueError("Input dictionary must contain a 'text' key with a string value for KeywordExtractor.")
        else:
            logger.error(f"KeywordExtractor: Invalid input data type. Expected str or dict, got {type(data)}.")
            raise ValueError(f"Input data must be a string or a dictionary with a 'text' key. Received type: {type(data)}")

        if not text_to_process.strip():
            logger.info("KeywordExtractor: Received empty or whitespace-only text for keyword extraction. Returning empty list.")
            return []

        # Convert to lowercase
        text_to_process = text_to_process.lower()

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text_no_punctuation = text_to_process.translate(translator)

        # Tokenize and filter
        all_words = text_no_punctuation.split()
        keywords_set: Set[str] = set()

        for word in all_words:
            if word and len(word) >= min_word_length and word not in stop_words:
                keywords_set.add(word)

        keywords = sorted(list(keywords_set)) # Sort for consistent output

        logger.info(f"KeywordExtractor: Extracted {len(keywords)} keywords. Sample: {keywords[:5]}...")
        return keywords