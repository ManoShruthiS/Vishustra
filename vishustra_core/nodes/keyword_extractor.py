import logging
import re
from typing import Any, Dict, List, Set
from collections import Counter

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node responsible for extracting significant keywords from text data.
    
    This node cleans the input text, removes common stop words, and returns 
    the most frequent terms based on a configurable threshold.
    """

    DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "at", "from", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "can", "could", "this", "that", "these", "those"
    }

    @property
    def node_name(self) -> str:
        return "KeywordExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the provided input data.
        
        Args:
            data (Any): The input text to process. Expected to be a string.
            context (Dict[str, Any]): Execution context, can contain 'top_n' 
                                      and 'exclude_words' overrides.
        
        Returns:
            List[str]: A list of extracted keywords.
            
        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type: {type(data)}. Expected string.")
            raise ValueError("KeywordExtractorNode requires string input data.")

        try:
            # Extraction parameters from context or defaults
            top_n = context.get("top_n", 10)
            custom_stop_words = context.get("exclude_words", set())
            stop_words = self.DEFAULT_STOP_WORDS.union(custom_stop_words)

            # Sanitize and tokenize
            # Remove punctuation and lowercase
            clean_text = re.sub(r'[^\w\s]', '', data.lower())
            words = clean_text.split()

            # Filter words
            filtered_words = [
                word for word in words 
                if word not in stop_words and len(word) > 2 and not word.isdigit()
            ]

            # Frequency analysis
            counts = Counter(filtered_words)
            keywords = [word for word, count in counts.most_common(top_n)]

            logger.debug(f"[{self.node_name}] Successfully extracted {len(keywords)} keywords.")
            return keywords

        except Exception as e:
            logger.exception(f"[{self.node_name}] Failed to process data: {str(e)}")
            raise e