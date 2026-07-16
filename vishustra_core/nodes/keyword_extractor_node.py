import logging
import re
from typing import Any, Dict, List, Set

# Assuming vishustra_core.nodes.base_node is correctly configured in the project's Python path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract keywords from a given text input.

    This node processes a string, normalizing it and then identifying
    unique words based on configurable criteria such as minimum length
    and a list of stopwords. The output is a sorted list of these
    extracted keywords.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords using a simple heuristic.

        The input `data` is expected to be a string. The `context` dictionary
        allows for customization of the extraction process, specifically
        for minimum keyword length and a custom set of stopwords.

        Args:
            data: The input text (string) from which keywords will be extracted.
            context: A dictionary containing runtime configuration and metadata.
                     Expected configurable keys include:
                     - 'min_length' (int, optional): The minimum character length
                                                    for a word to be considered a keyword.
                                                    Defaults to 3.
                     - 'stopwords' (List[str], optional): A list of words to be
                                                         excluded from the keyword list.
                                                         Defaults to a small, common English set.

        Returns:
            A sorted list of unique strings identified as keywords.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "KeywordExtractorNode received non-string input. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"KeywordExtractorNode expects string input, but received {type(data).__name__}"
            )

        # --- Configuration from context (with sensible defaults) ---
        min_length = context.get('min_length', 3)
        if not isinstance(min_length, int) or min_length < 1:
            logger.warning(
                "Invalid 'min_length' value in context: '%s'. Using default of 3.",
                min_length
            )
            min_length = 3

        # A small set of common English stopwords for demonstration
        default_stopwords: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "then", "else", "when", "where", "why",
            "how", "what", "which", "who", "whom", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "for", "at", "by", "with", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
        
        # Ensure custom stopwords are processed as a set for efficient lookup
        custom_stopwords = context.get('stopwords')
        if isinstance(custom_stopwords, list):
            stopwords: Set[str] = set(custom_stopwords)
        else:
            stopwords = default_stopwords
            if custom_stopwords is not None:
                logger.warning(
                    "Invalid 'stopwords' value in context. Expected 'list', got '%s'. Using default stopwords.",
                    type(custom_stopwords).__name__
                )

        text_lower = data.lower()
        
        # Use regex to find sequences of alphanumeric characters.
        # This effectively removes most punctuation and splits words.
        all_words = re.findall(r'\b[a-z0-9]+\b', text_lower)
        
        extracted_keywords: Set[str] = set()
        for word in all_words:
            if len(word) >= min_length and word not in stopwords:
                extracted_keywords.add(word)
                
        logger.info(
            "Successfully extracted %d unique keywords from input text using min_length=%d and %d stopwords.",
            len(extracted_keywords), min_length, len(stopwords)
        )
        
        # Return a sorted list for consistent and predictable output
        return sorted(list(extracted_keywords))
