import logging
from typing import Any, Dict, List, Set
import re

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A Vishustra processing node designed to extract keywords from text data.

    This node performs a simulated keyword extraction by processing an input
    string, converting it to lowercase, removing punctuation, filtering out
    common stopwords, and discarding very short words. The output is a list
    of unique potential keywords.

    In a production environment, this node would typically integrate with
    a dedicated NLP library (e.g., SpaCy, NLTK, or a more advanced LLM-based
    extraction service) to provide sophisticated keyword identification.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract keywords.

        The method expects a string as input. It converts the text to lowercase,
        tokenizes it, filters out a basic set of English stopwords, and
        discards words shorter than a predefined minimum length.

        Args:
            data: The input data, expected to be a string containing text
                  from which keywords are to be extracted.
            context: A dictionary containing operational context for the node.
                     Currently not utilized by this specific node's logic.

        Returns:
            A list of unique strings representing the extracted keywords,
            sorted alphabetically for consistent output. Returns an empty
            list if no keywords are found or if the input is an empty string.

        Raises:
            TypeError: If the input 'data' is not of type `str`.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractor received invalid input type. Expected 'str', "
                f"but got '{type(data).__name__}'. Data: {data!r}"
            )
            raise TypeError("KeywordExtractor requires string input for processing.")

        if not data.strip():
            logger.warning(
                "KeywordExtractor received an empty or whitespace-only string. "
                "Returning an empty list of keywords."
            )
            return []

        # Step 1: Normalize text to lowercase and tokenize by splitting on non-alphanumeric characters.
        # This also removes most punctuation.
        text = data.lower()
        words = [word for word in re.split(r'\W+', text) if word]

        # Step 2: Define a basic set of common English stopwords.
        # This set is intentionally minimal for simulation.
        stopwords = {
            "a", "an", "the", "and", "or", "is", "are", "was", "were", "of",
            "in", "on", "at", "for", "with", "to", "from", "by", "as", "it",
            "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
            "them", "this", "that", "those", "these", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "not", "but", "if",
            "then", "so", "what", "where", "when", "why", "how", "which",
            "who", "whom", "my", "your", "his", "her", "its", "our", "their"
        }

        # Step 3: Filter words based on stopwords and minimum length.
        # Use a set to automatically handle uniqueness.
        extracted_keywords: Set[str] = set()
        min_keyword_length = 3  # Configuration could make this dynamic

        for word in words:
            if word not in stopwords and len(word) >= min_keyword_length:
                extracted_keywords.add(word)

        # Step 4: Convert the set of unique keywords to a sorted list.
        result = sorted(list(extracted_keywords))

        logger.debug(f"Successfully extracted {len(result)} keywords from input data.")
        return result