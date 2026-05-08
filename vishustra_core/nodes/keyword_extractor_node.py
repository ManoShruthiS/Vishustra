import logging
import re
from typing import Any, Dict, List, Set
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract unique, meaningful keywords from string data.
    Filters out common stop words and non-alphabetic tokens to prepare data for 
    indexing or downstream LLM context injection.
    """

    # Basic stop words to filter out common noise
    DEFAULT_STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "at", "from", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "can", "could", "this", "that", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "my", "your", "his", "her", "its"
    }

    @property
    def node_name(self) -> str:
        """Returns the identifier for the keyword extraction node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input string to extract a list of keywords.

        Args:
            data: The raw input string to be processed.
            context: A dictionary containing runtime configuration. 
                     Supports 'top_n' (int) and 'extra_stop_words' (Set[str]).

        Returns:
            A list of unique keywords extracted from the text.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If an error occurs during tokenization or filtering.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type: expected str, got {type(data).__name__}")
            raise TypeError(f"{self.node_name} expects string input for keyword extraction.")

        try:
            # Extract configuration parameters from context
            top_n = context.get("top_n", 15)
            extra_stop_words = context.get("extra_stop_words", set())
            stop_words = self.DEFAULT_STOP_WORDS.union(extra_stop_words)

            # Normalization: Lowercase and strip non-alphanumeric characters
            # We use a regex to keep only alphanumeric characters and spaces
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', data.lower())
            
            # Simple tokenization by whitespace
            tokens = clean_text.split()

            # Filtering logic:
            # 1. Remove stop words
            # 2. Ignore short tokens (less than 3 characters)
            # 3. Ignore purely numeric tokens
            keywords: List[str] = []
            seen: Set[str] = set()

            for word in tokens:
                if (word not in stop_words and 
                    len(word) > 2 and 
                    not word.isdigit() and 
                    word not in seen):
                    keywords.append(word)
                    seen.add(word)

            # Limit result to requested top_n
            result = keywords[:top_n]

            logger.debug(f"[{self.node_name}] Extracted {len(result)} keywords from input length {len(data)}.")
            return result

        except Exception as e:
            logger.exception(f"[{self.node_name}] Unexpected error during keyword extraction: {str(e)}")
            raise ValueError(f"Keyword extraction failed: {str(e)}") from e