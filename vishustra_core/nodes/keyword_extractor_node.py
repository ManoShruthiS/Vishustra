import logging
import re
import string
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node that extracts keywords from a given text string.
    
    This node expects the input `data` to be a string. It processes the string
    by converting it to lowercase, removing punctuation, filtering out common
    stop words, and extracting unique words that are typically considered
    keywords.
    
    The `context` dictionary can optionally provide a custom list of stop words
    under the key 'stop_words'. If not provided, a default list is used.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input data.

        Args:
            data: The input data, expected to be a string containing text.
            context: A dictionary that can optionally contain 'stop_words' (List[str])
                     for custom stop word filtering.

        Returns:
            A list of unique keywords extracted from the text, sorted alphabetically.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "Invalid input data type for KeywordExtractorNode. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise ValueError("KeywordExtractorNode requires string input data.")

        if not data.strip():
            logger.debug("Input data is an empty string. Returning an empty list of keywords.")
            return []

        text = data.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        words = text.split()

        # Default list of common English stop words
        default_stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "at", "by", "for",
            "in", "on", "of", "to", "with", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "doing", "i", "me", "my",
            "you", "your", "he", "him", "his", "she", "her", "it", "its", "we", "us", "our",
            "they", "them", "their", "this", "that", "these", "those", "can", "could",
            "will", "would", "should", "may", "might", "must", "not", "no", "yes", "about",
            "above", "across", "after", "against", "along", "among", "around", "as", "before",
            "behind", "below", "beneath", "beside", "between", "beyond", "down", "during",
            "except", "inside", "into", "like", "near", "off", "out", "outside", "over",
            "past", "round", "since", "through", "under", "up", "upon", "while", "where",
            "when", "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too",
            "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
        
        # Allow stop words to be customized via context
        stop_words = set(context.get('stop_words', default_stop_words))

        # Filter out stop words and very short words (e.g., single letters after punctuation removal)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        unique_keywords = sorted(list(set(keywords)))
        
        logger.debug("Successfully extracted %d keywords: %s", len(unique_keywords), unique_keywords)
        return unique_keywords

