import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra processing node that extracts keywords from input text.
    It performs basic tokenization, stop-word removal, and minimum length
    filtering to identify relevant terms.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        The `data` is expected to be a string.
        The `context` dictionary can optionally provide:
        - 'stop_words' (Set[str]): A set of words to ignore during extraction.
          Defaults to a small common set if not provided.
        - 'min_word_length' (int): Minimum length for a word to be considered a keyword.
          Defaults to 3.
        - 'max_keywords' (int | None): Maximum number of keywords to return.
          If None, all extracted keywords are returned.

        Args:
            data: The input text (string) from which to extract keywords.
            context: A dictionary containing runtime parameters for the node.

        Returns:
            A sorted list of unique extracted keywords (strings).

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for KeywordExtractor. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError("KeywordExtractor expects 'data' to be a string.")

        text = data.lower()

        # Retrieve configuration from context or use defaults
        default_stop_words = {
            'a', 'an', 'the', 'is', 'of', 'and', 'in', 'for', 'to', 'with', 'on',
            'it', 'that', 'this', 'was', 'as', 'at', 'by', 'be', 'has', 'had'
        }
        stop_words: Set[str] = context.get('stop_words', default_stop_words)
        min_word_length: int = context.get('min_word_length', 3)
        max_keywords: int | None = context.get('max_keywords', None)

        # Simple tokenization: find all sequences of word characters
        words = re.findall(r'\b\w+\b', text)

        extracted_keywords: Set[str] = set()
        for word in words:
            if word not in stop_words and len(word) >= min_word_length:
                extracted_keywords.add(word)

        # Convert to list and sort for consistent output
        result_keywords = sorted(list(extracted_keywords))

        # Apply max_keywords limit if specified
        if max_keywords is not None and len(result_keywords) > max_keywords:
            result_keywords = result_keywords[:max_keywords]
            logger.debug(f"Truncated keywords to {max_keywords} as per context.")

        logger.info(f"Successfully extracted {len(result_keywords)} keywords.")
        return result_keywords