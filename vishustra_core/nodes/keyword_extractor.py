import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text string.

    This node performs a simple text processing routine to identify potential
    keywords by tokenizing the input, removing common stopwords, and filtering
    out non-alphanumeric tokens.
    """

    # A simple, illustrative set of common English stopwords.
    # In a production system, this would typically be loaded from a more
    # comprehensive library or configuration.
    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "if",
        "then", "else", "for", "with", "at", "from", "to", "in", "on", "of",
        "it", "its", "he", "she", "they", "we", "you", "your", "my", "our",
        "me", "him", "her", "us", "them", "this", "that", "these", "those",
        "have", "has", "had", "do", "does", "did", "not", "no", "yes", "can",
        "will", "would", "should", "could", "be", "been", "being", "am",
        "i", "you're", "i'm", "it's", "don't", "can't", "wouldn't", "must",
        "very", "much", "many", "few", "some", "any", "all", "each", "every",
        "such", "about", "above", "below", "between", "among", "through",
        "during", "before", "after", "again", "further", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now"
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract keywords.

        Args:
            data: The input text data as a string from which to extract keywords.
            context: A dictionary containing contextual information
                     (not directly used by this specific node, but required by BaseNode).

        Returns:
            A list of strings, where each string is an extracted keyword.
            Returns an empty list if no keywords are found or if the input is invalid.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"Invalid input type for KeywordExtractorNode. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.warning("Input data is empty or only whitespace. No keywords to extract.")
            return []

        text = data.lower()
        # Tokenize the text, keeping only alphanumeric characters and splitting by non-alphanumeric.
        words = re.findall(r'\b[a-z0-9]+\b', text)

        extracted_keywords: List[str] = []
        for word in words:
            # Filter out short words and stopwords
            if len(word) > 2 and word not in self._STOP_WORDS:
                extracted_keywords.append(word)

        # Return unique keywords to avoid duplicates from simple splitting
        unique_keywords = sorted(list(set(extracted_keywords)))

        if not unique_keywords:
            logger.info("No meaningful keywords extracted from the input text.")
        else:
            logger.debug(f"Successfully extracted {len(unique_keywords)} keywords.")

        return unique_keywords

if __name__ == '__main__':
    # Simple demonstration of the node's functionality
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    extractor = KeywordExtractorNode()
    
    test_texts = [
        "Vishustra is a highly modular LLM orchestration framework in Python.",
        "The quick brown fox jumps over the lazy dog.",
        "A an the is are was were.", # Should return empty or very few
        "",
        123, # Invalid type
        "  This is an important concept.  "
    ]

    print(f"Node Name: {extractor.node_name}")
    print("-" * 30)

    for i, text_input in enumerate(test_texts):
        print(f"\nTest Case {i+1}: Input '{text_input}'")
        try:
            keywords = extractor.process(text_input, {})
            print(f"Extracted Keywords: {keywords}")
        except TypeError as e:
            print(f"Error processing: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    print("-" * 30)
