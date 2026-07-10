import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract keywords from textual data.

    This node expects the input `data` to be a string. It performs a basic
    keyword extraction process by tokenizing the text, sanitizing it by
    removing non-alphabetic characters, converting all words to lowercase,
    and then filtering words based on a minimum length threshold.
    The output is a sorted list of unique keywords.

    Context Parameters:
    - 'min_word_length' (int, optional): The minimum character length a word
      must have to be considered a keyword. Defaults to 3 if not provided.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Executes the keyword extraction process on the input data.

        Args:
            data: The input payload, expected to be a string containing the text
                  from which keywords are to be extracted.
            context: A dictionary containing runtime parameters for the node.
                     Supports 'min_word_length' (int) to customize filtering.

        Returns:
            A sorted list of unique strings, each representing an extracted keyword.

        Raises:
            TypeError: If the 'data' input is not a string.
            ValueError: If 'min_word_length' in the context is not a positive integer.
        """
        if not isinstance(data, str):
            logger.error(
                "KeywordExtractorNode received invalid data type. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"KeywordExtractorNode expects string data, but received {type(data).__name__}."
            )

        # Handle empty or whitespace-only strings gracefully
        if not data.strip():
            logger.info("KeywordExtractorNode received an empty or whitespace-only string. Returning an empty list.")
            return []

        # Retrieve min_word_length from context or use default
        min_word_length = context.get("min_word_length", 3)
        if not isinstance(min_word_length, int) or min_word_length <= 0:
            logger.error(
                "Invalid 'min_word_length' value in context: '%s'. Must be a positive integer.",
                min_word_length
            )
            raise ValueError(
                f"'min_word_length' in context must be a positive integer, got {min_word_length}."
            )

        logger.debug(
            "Processing data with min_word_length=%d. Data snippet: '%s...'",
            min_word_length,
            data[:100] # Log a snippet for debug context
        )

        # Convert text to lowercase and use regex to find all sequences of alphabetic characters.
        # This effectively removes punctuation and splits words.
        words = re.findall(r'\b[a-zA-Z]+\b', data.lower())

        extracted_keywords = set()
        for word in words:
            if len(word) >= min_word_length:
                extracted_keywords.add(word)

        # Convert the set of unique keywords to a sorted list
        result_keywords = sorted(list(extracted_keywords))

        logger.debug("Successfully extracted %d unique keywords.", len(result_keywords))
        return result_keywords