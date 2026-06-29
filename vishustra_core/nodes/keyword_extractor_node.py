import logging
from typing import Any, Dict, List, Set

# Assuming BaseNode is located at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node responsible for extracting keywords from textual data.

    This implementation provides a basic, simulated keyword extraction by tokenizing
    the input text, converting words to lowercase, filtering out common stop words,
    and removing non-alphanumeric characters. It's intended for demonstration
    and acts as a placeholder for more sophisticated NLP-based keyword extraction
    techniques in a production environment.
    """

    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "is", "and", "or", "in", "of", "to", "for", "with", "on", "it",
        "as", "by", "from", "at", "be", "was", "were", "are", "been", "have", "has", "had",
        "do", "does", "did", "not", "but", "so", "if", "then", "this", "that", "these",
        "those", "my", "your", "his", "her", "its", "our", "their", "we", "you", "he",
        "she", "they", "i", "me", "him", "us", "them", "which", "what", "where", "when",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "don't", "should", "could"
    }
    _MIN_KEYWORD_LENGTH: int = 3

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data, expecting a string, to extract a list of keywords.

        Args:
            data: The input text from which keywords are to be extracted.
                  Expected type is 'str'.
            context: A dictionary containing additional runtime context relevant
                     to the current processing operation.

        Returns:
            A sorted list of unique strings representing the extracted keywords.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string or contains only
                        whitespace characters.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name}: Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg, extra={'node': self.node_name, 'input_type': type(data).__name__, 'context': context})
            raise TypeError(error_msg)

        text_content = data.strip()

        if not text_content:
            error_msg = f"{self.node_name}: Input text cannot be empty or consist only of whitespace."
            logger.warning(error_msg, extra={'node': self.node_name, 'context': context})
            raise ValueError(error_msg)

        logger.debug(
            f"{self.node_name}: Starting keyword extraction for text (first 100 chars): "
            f"'{text_content[:100]}{'...' if len(text_content) > 100 else ''}'",
            extra={'node': self.node_name, 'context': context}
        )

        words = text_content.lower().split()
        extracted_keywords: Set[str] = set()

        for word in words:
            # Remove non-alphanumeric characters from the word
            clean_word = ''.join(char for char in word if char.isalnum())

            if (clean_word and
                    len(clean_word) >= self._MIN_KEYWORD_LENGTH and
                    clean_word not in self._STOP_WORDS):
                extracted_keywords.add(clean_word)

        result_keywords = sorted(list(extracted_keywords))
        logger.info(
            f"{self.node_name}: Successfully extracted {len(result_keywords)} keywords.",
            extra={'node': self.node_name, 'keyword_count': len(result_keywords), 'context': context}
        )
        return result_keywords