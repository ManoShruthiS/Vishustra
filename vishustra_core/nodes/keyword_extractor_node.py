import logging
import re
from typing import Any, Dict, List, Set
from collections import Counter

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract high-frequency keywords from text data.
    It filters out common stop words and uses regex-based tokenization to ensure 
    only meaningful alphanumeric tokens are preserved.
    """

    DEFAULT_STOP_WORDS: Set[str] = {
        "the", "is", "at", "which", "on", "and", "a", "an", "to", "in", "for", 
        "of", "with", "as", "by", "from", "it", "that", "this", "be", "are", 
        "was", "were", "has", "have", "had"
    }

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for the keyword extraction node.
        """
        return "KeywordExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the provided text data.

        Args:
            data: The input text (expected to be a string).
            context: A dictionary containing configuration overrides like 'top_k' 
                     or 'custom_stop_words'.

        Returns:
            A list of top_k strings representing the most frequent keywords.

        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected str, got {type(data).__name__}.")
            raise ValueError("KeywordExtractorNode requires string input data.")

        try:
            top_k = context.get("top_k", 5)
            custom_stop_words = context.get("stop_words", self.DEFAULT_STOP_WORDS)
            
            # Clean and tokenize the text
            cleaned_text = data.lower()
            tokens = re.findall(r'\b\w{3,}\b', cleaned_text)
            
            # Filter tokens
            filtered_tokens = [
                token for token in tokens 
                if token not in custom_stop_words and not token.isdigit()
            ]
            
            # Count frequencies
            counts = Counter(filtered_tokens)
            keywords = [word for word, count in counts.most_common(top_k)]
            
            logger.info(f"[{self.node_name}] Successfully extracted {len(keywords)} keywords from input.")
            return keywords

        except Exception as e:
            logger.exception(f"[{self.node_name}] Failed to process keywords: {str(e)}")
            raise e