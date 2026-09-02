import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class KeywordExtractor(BaseNode):
    """
    A Vishustra processing node that extracts keywords from input text data.

    This node simulates keyword extraction by identifying unique words that
    meet a minimum length requirement. It normalizes the text by converting
    to lowercase and removing punctuation before splitting into words.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data based on a minimum word length.

        The process involves:
        1. Validating that the input `data` is a string.
        2. Retrieving configuration for `min_word_length` from the `context`,
           defaulting to 3 if not specified or invalid.
        3. Normalizing the text (lowercase, remove non-alphanumeric characters).
        4. Splitting the normalized text into words.
        5. Filtering words based on `min_word_length`.
        6. Returning a sorted list of unique extracted keywords.

        Args:
            data: The input data, expected to be a string containing text.
            context: A dictionary containing runtime configuration.
                     Expected keys:
                     - 'min_word_length' (int, optional): The minimum length
                       a word must have to be considered a keyword. Defaults to 3.

        Returns:
            A list of unique keywords (strings), sorted alphabetically.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for {self.node_name}. "
                f"Expected str, got {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Retrieve min_word_length from context or use a default
        min_word_length = context.get("min_word_length", 3)
        if not isinstance(min_word_length, int) or min_word_length <= 0:
            logger.warning(
                f"Invalid 'min_word_length' in context for {self.node_name}: '{min_word_length}'. "
                "Falling back to default of 3."
            )
            min_word_length = 3

        logger.info(
            f"[{self.node_name}] Starting keyword extraction with min_word_length={min_word_length}."
        )

        # Normalize text: convert to lowercase and remove non-alphanumeric characters
        # Using re.sub to replace anything not a letter, number, or whitespace with a space,
        # then splitting on whitespace. This helps handle punctuation attached to words.
        normalized_text = re.sub(r"[^a-z0-9\s]", " ", data.lower())

        # Split into words and filter by length
        # Using a set to automatically handle uniqueness
        words = normalized_text.split()
        keywords = {word for word in words if len(word) >= min_word_length}

        # Convert back to list and sort for deterministic output
        sorted_keywords = sorted(list(keywords))

        logger.debug(
            f"[{self.node_name}] Extracted {len(sorted_keywords)} unique keywords."
        )
        return sorted_keywords
