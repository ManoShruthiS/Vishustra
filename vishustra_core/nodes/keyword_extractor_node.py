import logging
import re
from collections import Counter
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node that extracts keywords from a given text.

    This node tokenizes the input text, removes common stop words and punctuation,
    and then identifies the most frequent remaining words as keywords.
    The number of keywords to extract can be configured via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract keywords.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        from which to extract keywords.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Can include 'num_keywords' (int) to specify
                                      how many keywords to return (defaults to 5).

        Returns:
            List[str]: A list of extracted keywords, ordered by frequency.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error("KeywordExtractorNode received non-string data. Type: %s", type(data))
            raise ValueError("KeywordExtractorNode expects string data for keyword extraction. "
                             f"Received type: {type(data)}.")

        if not data.strip():
            logger.warning("KeywordExtractorNode received empty or whitespace-only string, returning empty list.")
            return []

        text = data.lower()
        
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text) # Remove digits

        words = text.split()

        # A basic set of stop words for general English text.
        # For more advanced scenarios, a dedicated NLP library would provide a more comprehensive list.
        stop_words = {
            "a", "an", "the", "is", "are", "and", "or", "in", "on", "at", "for", "with", "to", "of", 
            "it", "this", "that", "i", "you", "he", "she", "we", "they", "be", "has", "have", "do", 
            "can", "will", "would", "should", "not", "but", "by", "from", "as", "if", "then", "its", 
            "our", "your", "my", "me", "him", "her", "us", "them", "which", "what", "when", "where", 
            "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
            "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "m", "d", "ll", 
            "ve", "re", "just", "don", "now", "about", "above", "after", "again", "against", "ain", 
            "among", "amongst", "around", "before", "behind", "below", "beneath", "beside", "between", 
            "beyond", "etc"
        }
        
        # Filter out stop words and short words (length <= 2, to avoid single letters or common acronyms)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        if not filtered_words:
            logger.warning("After filtering, no significant words found in the input text for keyword extraction. "
                           "Input sample: '%s...'", data[:100].replace('\n', ' '))
            return []

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Determine the number of keywords to extract, default to 5 if not in context
        num_keywords = context.get("num_keywords", 5)
        if not isinstance(num_keywords, int) or num_keywords <= 0:
            logger.warning("Invalid 'num_keywords' in context (%s). Defaulting to 5.", num_keywords)
            num_keywords = 5
        
        keywords = [word for word, count in word_counts.most_common(num_keywords)]

        logger.debug("Extracted keywords: %s from text sample: '%s...'", keywords, data[:50].replace('\n', ' '))
        return keywords
