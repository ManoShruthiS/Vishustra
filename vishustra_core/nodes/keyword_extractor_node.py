import logging
from typing import Any, Dict, List, Set
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract keywords from an input text string.

    This node simulates the keyword extraction process by tokenizing the input text,
    converting tokens to lowercase, filtering out common stop words, and
    returning a unique, sorted list of significant words. It provides robust
    handling for various input conditions and leverages logging for operational insights.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node, "Keyword Extractor".
        """
        return "Keyword Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data, expected to be a string, to identify and extract keywords.

        The method performs the following steps:
        1. Validates the input `data` type, ensuring it is a string.
        2. Strips whitespace from the text and handles empty inputs gracefully.
        3. Converts the text to lowercase.
        4. Splits the text into individual words (tokens).
        5. Filters out common stop words and non-alphabetic characters from tokens.
        6. Considers only words with a length greater than 2 characters.
        7. Collects unique significant words and returns them as a sorted list.

        Args:
            data (Any): The input data, anticipated to be a string containing the text
                        from which keywords are to be extracted.
            context (Dict[str, Any]): A dictionary providing contextual information or
                                       configuration. This can be used to pass custom
                                       `stop_words` (a `Set[str]`) to override the
                                       default set.

        Returns:
            List[str]: A list of unique keywords extracted from the input text,
                       sorted alphabetically. Returns an empty list if no keywords
                       are found or if the input text is empty.

        Raises:
            ValueError: If the input `data` is not a string.
            Exception: Captures and re-raises any other unexpected errors that occur
                       during the keyword extraction process, logging the full traceback.
        """
        logger.debug(f"[{self.node_name}] Starting keyword extraction for input data.")

        if not isinstance(data, str):
            error_msg = (f"[{self.node_name}] Invalid input data type. "
                         f"Expected 'str', but received '{type(data).__name__}'.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        text = data.strip()
        if not text:
            logger.warning(f"[{self.node_name}] Input text is empty after stripping; returning an empty list.")
            return []

        # Default stop words for demonstration purposes.
        # In a production environment, this might be loaded from a configuration file,
        # a database, or a more sophisticated NLP library.
        default_stop_words: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for", "nor", "so",
            "yet", "at", "by", "in", "on", "of", "to", "with", "from", "as", "it", "its", "he", "she",
            "him", "her", "they", "them", "we", "us", "you", "your", "that", "this", "these", "those",
            "i", "me", "my", "mine", "our", "ours", "their", "theirs", "what", "when", "where", "why",
            "how", "who", "whom", "which", "whose", "if", "then", "else", "over", "under", "about",
            "above", "below", "before", "after", "again", "further", "once", "here", "there",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "not",
            "only", "own", "same", "than", "too", "very", "s", "t", "can", "will", "just",
            "don", "should", "now", "ve", "ll", "re", "m", "d", "has", "have", "had", "do", "does", "did",
            "be", "been", "being"
        }
        stop_words: Set[str] = context.get('stop_words', default_stop_words)

        try:
            # Tokenize, convert to lowercase, and filter tokens.
            words = text.lower().split()
            extracted_keywords: Set[str] = set()
            for word in words:
                # Remove non-alphabetic characters from the word
                cleaned_word = ''.join(filter(str.isalpha, word))
                # Add to set if it's not a stop word and has a meaningful length
                if cleaned_word and cleaned_word not in stop_words and len(cleaned_word) > 2:
                    extracted_keywords.add(cleaned_word)

            sorted_keywords = sorted(list(extracted_keywords))
            logger.info(f"[{self.node_name}] Successfully extracted {len(sorted_keywords)} unique keywords.")
            logger.debug(f"[{self.node_name}] Extracted keywords: {sorted_keywords}")
            return sorted_keywords
        except Exception as e:
            error_msg = (f"[{self.node_name}] An unexpected error occurred during "
                         f"keyword extraction: {e}")
            logger.exception(error_msg) # Log the exception with traceback
            raise # Re-raise the exception after logging for upstream handling.