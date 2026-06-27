import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A processing node designed to extract keywords from a given text input.
    This node performs a simulated keyword extraction by tokenizing the text,
    converting to lowercase, filtering based on word length, and optionally
    excluding a set of stop words provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts a list of unique keywords from the input text data.

        The extraction process involves:
        1. Validating the input data type to ensure it's a string.
        2. Converting the text to lowercase.
        3. Tokenizing the text into individual words, stripping punctuation.
        4. Filtering words based on a minimum length specified in the context.
        5. Filtering out stop words, if a set of stop words is provided in the context.
        6. Ensuring only unique keywords are returned.

        Args:
            data (Any): The input data expected to be a string (text) from which
                        keywords are to be extracted.
            context (Dict[str, Any]): A dictionary containing configuration parameters
                                      for the extraction process.
                                      Expected keys:
                                      - 'stop_words' (Set[str], optional): A set of words
                                        to be ignored during keyword extraction. Defaults to an
                                        empty set if not provided or invalid.
                                      - 'min_keyword_length' (int, optional): The minimum
                                        character length a word must have to be considered
                                        a keyword. Defaults to 3 if not provided or invalid.

        Returns:
            List[str]: A list of unique keywords extracted from the input text.
                       Returns an empty list if the input data is not valid or
                       if no keywords are found after filtering.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'min_keyword_length' provided in the context is not
                        a positive integer.
        """
        logger.debug(f"[{self.node_name}] Initiating processing for data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' node must be a string, "
                f"got '{type(data).__name__}'."
            )

        text = data.lower()

        # Retrieve and validate 'stop_words' from context
        stop_words: Set[str] = context.get('stop_words', set())
        if not isinstance(stop_words, set):
            logger.warning(
                f"[{self.node_name}] 'stop_words' in context is not a set. "
                "Proceeding without a custom stop-word list."
            )
            stop_words = set()

        # Retrieve and validate 'min_keyword_length' from context
        min_keyword_length: int = context.get('min_keyword_length', 3)
        if not isinstance(min_keyword_length, int) or min_keyword_length <= 0:
            logger.error(
                f"[{self.node_name}] Invalid 'min_keyword_length' in context: '{min_keyword_length}'. "
                "Must be a positive integer."
            )
            raise ValueError(
                f"'min_keyword_length' must be a positive integer, "
                f"got '{min_keyword_length}' in context."
            )
        
        # Tokenize the text using a simple regex to capture word boundaries
        # and filter out any empty strings resulting from the split.
        words = [word for word in re.findall(r'\b\w+\b', text) if word]

        extracted_keywords: List[str] = []
        seen_keywords: Set[str] = set() # Use a set for efficient uniqueness check

        for word in words:
            if (len(word) >= min_keyword_length and
                word not in stop_words and
                word not in seen_keywords):
                extracted_keywords.append(word)
                seen_keywords.add(word)
        
        logger.debug(
            f"[{self.node_name}] Successfully extracted {len(extracted_keywords)} keywords. "
            f"Example keywords: {extracted_keywords[:5]}..."
        )
        return extracted_keywords