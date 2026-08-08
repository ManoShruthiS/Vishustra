import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node that extracts keywords from a given text.

    This node performs a basic keyword extraction by:
    1. Converting input text to lowercase.
    2. Removing punctuation and numeric characters.
    3. Tokenizing the text into words.
    4. Filtering out common English stop words.
    5. Filtering out words shorter than a configurable minimum length.
    6. Returning a sorted list of unique keywords.
    
    The stop word list is internal but the minimum word length can be
    overridden via the `context` dictionary.
    """

    # A sensible default list of common English stop words.
    # For performance, this is a set.
    _STOP_WORDS = {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "else", "when", "where", "how", "why",
        "for", "at", "by", "with", "from", "to", "of", "in", "on", "out", "up", "down",
        "this", "that", "these", "those", "it", "its", "he", "she", "i", "we", "you", "they",
        "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
        "me", "himself", "herself", "myself", "yourself", "ourselves", "themselves",
        "what", "which", "who", "whom", "whose", "here", "there", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "about", "above", "after", "again", "against", "among", "below", "between", "had", "has",
        "have", "having", "do", "does", "did", "doing", "would", "could", "shall", "may", "must",
        "ought", "wouldn", "couldn", "shouldn", "isn", "aren", "wasn", "weren", "hasn", "haven",
        "hadn", "won", "wouldn", "can't", "don't", "doesn't", "didn't", "can't", "couldn't",
        "shouldn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
        "won't", "wouldn't"
    }
    
    _DEFAULT_MIN_WORD_LENGTH = 3

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data.

        Args:
            data: The input text as a string from which keywords are to be extracted.
            context: A dictionary containing contextual information.
                     Can be used to override 'min_keyword_length' (int)
                     for custom minimum word length filtering.

        Returns:
            A sorted list of unique keywords extracted from the text.

        Raises:
            ValueError: If the input 'data' is not a string, or if an unexpected
                        issue prevents successful keyword extraction.
        """
        logger.info(f"[{self.node_name}] Starting keyword extraction process.")
        
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, but received {type(data).__name__}."
            )
            raise ValueError(
                f"{self.node_name} requires string input, "
                f"but received {type(data).__name__}."
            )
        
        try:
            # Retrieve configurable minimum word length from context or use default
            min_length = context.get('min_keyword_length', self._DEFAULT_MIN_WORD_LENGTH)
            if not isinstance(min_length, int) or min_length < 1:
                logger.warning(
                    f"[{self.node_name}] Invalid 'min_keyword_length' in context "
                    f"({min_length}). Using default: {self._DEFAULT_MIN_WORD_LENGTH}."
                )
                min_length = self._DEFAULT_MIN_WORD_LENGTH

            # Convert to lowercase
            text = data.lower()

            # Remove punctuation and numbers, replace with a single space to avoid
            # concatenating words that were separated by punctuation.
            cleaned_text = re.sub(r'[^a-z\s]', ' ', text)
            
            # Tokenize words by splitting on whitespace
            words = cleaned_text.split()

            # Filter out stop words and words shorter than the minimum length
            filtered_words = [
                word for word in words 
                if word not in self._STOP_WORDS and len(word) >= min_length
            ]

            # Get unique keywords and sort them for consistent output
            unique_keywords = sorted(list(set(filtered_words)))
            
            logger.info(
                f"[{self.node_name}] Successfully extracted {len(unique_keywords)} "
                f"unique keywords (min_length={min_length})."
            )
            return unique_keywords
        
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during "
                f"keyword extraction: {e}"
            )
            # Re-raise the exception to propagate the error up the call stack
            raise ValueError(f"Failed to extract keywords due to an internal error: {e}") from e
