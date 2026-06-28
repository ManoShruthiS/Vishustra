import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract keywords from a given text.

    This node expects a string as its input data and attempts to identify
    and return a list of relevant keywords. For this initial implementation,
    it performs a basic tokenization, converts text to lowercase, removes
    common stop words and very short words, and strips basic punctuation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data.

        Args:
            data: The input text from which keywords need to be extracted.
                  Expected to be a string.
            context: A dictionary containing contextual information relevant
                     to the current orchestration run. Not directly used
                     for keyword extraction logic in this basic version.

        Returns:
            A list of strings, where each string is an extracted keyword.
            Returns an empty list if no significant keywords are found or
            if the input is an empty string.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name} expects input data of type 'str', "
                f"but received {type(data).__name__}. Aborting keyword extraction."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle empty or whitespace-only strings gracefully
        if not data.strip():
            logger.info(
                f"{self.node_name} received an empty or whitespace-only string. "
                "Returning an empty list of keywords."
            )
            return []

        # --- Simulate keyword extraction logic ---
        # For a production-grade system, this would typically involve:
        # - Advanced NLP libraries (e.g., NLTK, spaCy)
        # - LLM calls for sophisticated keyword generation
        # - Domain-specific dictionaries or models

        text_lower = data.lower()
        # Basic tokenization: split by spaces
        raw_words = text_lower.split()

        # Simple list of common English stop words and minimum word length filter
        stop_words = {
            "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
            "for", "with", "at", "by", "from", "into", "of", "on", "to", "up", "down",
            "in", "out", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
        min_keyword_length = 3  # Minimum length for a word to be considered a keyword

        extracted_keywords = []
        seen_keywords = set() # To ensure unique keywords

        for word in raw_words:
            # Strip common punctuation from the beginning and end of the word
            cleaned_word = word.strip('.,!?;:\"\'()[]{}<>*-+/=_`~@#$%^&|\\')

            if cleaned_word and cleaned_word not in stop_words and len(cleaned_word) >= min_keyword_length:
                if cleaned_word not in seen_keywords:
                    extracted_keywords.append(cleaned_word)
                    seen_keywords.add(cleaned_word)

        logger.debug(
            f"{self.node_name} successfully processed data. "
            f"Extracted {len(extracted_keywords)} keywords: {extracted_keywords}"
        )
        return extracted_keywords