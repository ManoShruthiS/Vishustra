import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that simulates keyword extraction from text data.

    This node is designed to take a string as input, process it to identify
    potential keywords based on configurable criteria, and return them as a list.
    It supports basic text normalization and filtering based on word length.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        The keyword extraction process involves:
        1. Validating the input data type (must be a string).
        2. Normalizing the text (converting to lowercase, removing non-alphanumeric characters).
        3. Filtering words based on a minimum length, configurable via the context.
        4. Optionally ensuring that only unique keywords are returned, also configurable.

        Args:
            data: The input data, expected to be a string containing the text
                  from which keywords should be extracted.
            context: A dictionary containing operational context and configuration.
                     Recognized keys include:
                     - 'min_keyword_length' (int): Minimum length for a word to be
                       considered a keyword. Defaults to 3.
                     - 'unique_keywords_only' (bool): If True, returns only unique
                       keywords. Defaults to True.

        Returns:
            A list of strings, where each string is an extracted keyword.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'min_keyword_length' in the context is not a positive integer.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for KeywordExtractorNode. "
                f"Expected str, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Retrieve configuration from context with sensible defaults
        min_keyword_length = context.get('min_keyword_length', 3)
        unique_keywords_only = context.get('unique_keywords_only', True)

        if not isinstance(min_keyword_length, int) or min_keyword_length <= 0:
            error_msg = (
                f"Invalid 'min_keyword_length' configuration in context. "
                f"Expected a positive integer, but got {min_keyword_length}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(
            f"KeywordExtractorNode processing text (length: {len(data)}) with "
            f"min_length={min_keyword_length}, unique_only={unique_keywords_only}."
        )

        # Normalize and tokenize the text
        # Convert to lowercase and split by non-alphanumeric sequences
        normalized_text = data.lower()
        words = re.findall(r'\b[a-z0-9]+\b', normalized_text) # Extracts alphanumeric words

        # Filter keywords based on minimum length
        extracted_keywords = [
            word for word in words
            if len(word) >= min_keyword_length
        ]

        if unique_keywords_only:
            # Sort for deterministic output if unique words are requested
            extracted_keywords = sorted(list(set(extracted_keywords)))

        logger.debug(f"KeywordExtractorNode extracted {len(extracted_keywords)} keywords.")
        return extracted_keywords
