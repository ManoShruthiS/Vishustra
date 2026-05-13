import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node responsible for extracting significant keywords from text data.
    Uses a frequency-based approach with stop-word filtering and regex tokenization.
    """

    DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from", 
        "by", "for", "with", "about", "against", "between", "into", "through", "during", 
        "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
        "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
        "t", "can", "will", "just", "don", "should", "now", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "doing", "do", "does"
    }

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input string to extract top keywords.
        
        Args:
            data: The input text to analyze (expected type: str).
            context: Dictionary containing configuration like 'top_k' or 'min_word_length'.
            
        Returns:
            A dictionary containing the extracted keywords and their frequencies.
        """
        logger.info("Starting keyword extraction process.")

        if not isinstance(data, str):
            error_msg = f"Invalid data type received: {type(data)}. Expected str."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.warning("Empty string provided to KeywordExtractorNode.")
            return {"keywords": [], "metadata": {"status": "empty_input"}}

        try:
            # Extraction parameters from context
            top_k = context.get("top_k", 10)
            min_length = context.get("min_word_length", 3)
            custom_stop_words = context.get("stop_words", set())
            
            combined_stop_words = self.DEFAULT_STOP_WORDS.union(custom_stop_words)

            # Tokenization: lowercasing and removing non-alphanumeric characters
            words = re.findall(r'\b\w+\b', data.lower())
            
            # Filtering based on length and stop words
            filtered_words = [
                word for word in words 
                if len(word) >= min_length and word not in combined_stop_words
            ]

            # Frequency analysis
            counts = Counter(filtered_words)
            top_keywords = counts.most_common(top_k)

            logger.info(f"Successfully extracted {len(top_keywords)} keywords.")
            
            return {
                "keywords": [
                    {"word": word, "count": count} 
                    for word, count in top_keywords
                ],
                "metadata": {
                    "total_tokens": len(words),
                    "filtered_tokens": len(filtered_words),
                    "top_k": top_k
                }
            }

        except Exception as e:
            logger.exception("An error occurred during keyword extraction.")
            return {
                "keywords": [],
                "metadata": {"status": "error", "error_detail": str(e)}
            }