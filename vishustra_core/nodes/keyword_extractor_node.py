import logging
import re
from typing import Any, Dict, List, Set

# Assuming BaseNode is available at this path as per project instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed for extracting keywords from textual data.

    This node provides a simulated keyword extraction mechanism by performing
    the following steps:
    1. Validating input to ensure it is a string.
    2. Converting the text to lowercase.
    3. Tokenizing the text into words, removing punctuation.
    4. Filtering out common stopwords (configurable via context).
    5. Filtering out words shorter than a specified minimum length (configurable via context).
    6. Returning a sorted list of unique extracted keywords.

    Configuration parameters for stopwords and minimum word length can be
    provided through the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract relevant keywords.

        The method expects `data` to be a string (e.g., a document, paragraph,
        or sentence). It leverages regular expressions for robust tokenization
        and employs configurable lists for stopword filtering and minimum word
        length criteria to refine the output.

        Args:
            data (Any): The input data from which keywords are to be extracted.
                        Must be a string.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       or configuration parameters.
                                       Accepted keys:
                                       - 'stopwords' (List[str]): An iterable of
                                         words to be ignored. Defaults to a common
                                         English stopword list if not provided.
                                       - 'min_word_length' (int): The minimum
                                         character length for a word to be
                                         considered a keyword. Defaults to 3.

        Returns:
            List[str]: A sorted list of unique strings representing the extracted keywords.

        Raises:
            ValueError: If the input `data` is not a string, indicating an
                        incorrect data type for this node's operation.
            Exception: Catches and re-raises any other unforeseen errors that
                       occur during the keyword extraction process, logging
                       the full traceback for diagnostic purposes.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type. Expected 'str' for keyword "
                f"extraction, but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # --- Configuration from context or defaults ---
            # A default, simplified set of common English stopwords
            default_stopwords: Set[str] = {
                "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                "and", "or", "but", "for", "nor", "so", "yet", "to", "of", "in",
                "on", "at", "by", "with", "from", "into", "during", "through",
                "about", "against", "between", "before", "after", "above", "below",
                "up", "down", "out", "off", "over", "under", "again", "further",
                "then", "once", "here", "there", "when", "where", "why", "how",
                "all", "any", "both", "each", "few", "more", "most", "other",
                "some", "such", "no", "not", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
            }
            # Allow context to override default stopwords
            stopwords: Set[str] = set(context.get('stopwords', default_stopwords))
            # Allow context to override default minimum word length
            min_word_length: int = context.get('min_word_length', 3)

            logger.debug(
                f"[{self.node_name}] Initiating keyword extraction for input data. "
                f"Config: min_word_length={min_word_length}, {len(stopwords)} stopwords loaded."
            )

            # --- Text Preprocessing ---
            text_lower = data.lower()
            
            # Use regex to find all sequences of alphabetic characters, effectively
            # removing punctuation and splitting into words.
            words = re.findall(r'\b[a-z]+\b', text_lower)

            # --- Keyword Filtering ---
            extracted_keywords: List[str] = []
            for word in words:
                # Filter based on length and stopword list
                if len(word) >= min_word_length and word not in stopwords:
                    extracted_keywords.append(word)

            # Ensure uniqueness and return a sorted list for consistent output
            unique_keywords = sorted(list(set(extracted_keywords)))
            
            logger.info(
                f"[{self.node_name}] Successfully extracted {len(unique_keywords)} "
                f"unique keywords from the input data."
            )
            logger.debug(f"[{self.node_name}] Extracted keywords: {unique_keywords}")

            return unique_keywords

        except Exception as e:
            # Catch all other exceptions and log with traceback before re-raising
            error_msg = f"[{self.node_name}] An unexpected error occurred during keyword extraction."
            logger.exception(error_msg) # This logs the exception details including traceback
            raise # Re-raise the original exception
