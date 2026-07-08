import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract unique, relevant keywords from input text.

    This node tokenizes the input text, cleans words (e.g., removes punctuation,
    converts to lowercase), and filters them based on configurable parameters
    like minimum length and a list of stop words provided in the context.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts a sorted list of unique keywords from the provided text data.

        The 'data' input is expected to be a string. The 'context' dictionary
        allows for dynamic configuration of the extraction process.

        Configuration parameters in 'context':
        - 'min_keyword_length' (int, optional): The minimum character length
          a word must have to be considered a keyword. Defaults to 3.
        - 'stop_words' (List[str], optional): A list of words to be excluded
          from the keyword set (case-insensitive). Defaults to an empty list.

        Args:
            data: The input text from which keywords are to be extracted.
                  Expected to be a string.
            context: A dictionary containing runtime configuration and parameters
                     for the node's operation.

        Returns:
            A sorted list of unique keywords extracted from the text. Returns
            an empty list if no keywords are found or if the input text is empty.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractorNode received invalid input type. Expected 'str', "
                f"but got '{type(data).__name__}'. Unable to process."
            )
            raise TypeError("KeywordExtractorNode requires 'data' to be a string.")

        if not data.strip():
            logger.info("Received an empty or whitespace-only string for keyword extraction. Returning an empty list.")
            return []

        # Retrieve configuration from context with sensible defaults
        min_keyword_length = context.get("min_keyword_length", 3)
        stop_words_raw: List[str] = context.get("stop_words", [])
        # Convert stop words to a set of lowercase words for efficient, case-insensitive lookup
        stop_words: Set[str] = {word.lower() for word in stop_words_raw}

        # Normalize text to lowercase to ensure case-insensitive processing
        text_lower = data.lower()

        # Tokenize the text: find sequences of word characters (alphanumeric + underscore)
        # This regex effectively splits text into words while stripping most punctuation
        words = re.findall(r"\b\w+\b", text_lower)

        extracted_keywords: Set[str] = set()
        for word in words:
            if len(word) >= min_keyword_length and word not in stop_words:
                extracted_keywords.add(word)

        # Convert the set of keywords to a sorted list for consistent output
        return sorted(list(extracted_keywords))