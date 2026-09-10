import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node responsible for extracting keywords from textual data.
    This node performs a basic tokenization and filtering process to identify
    potential keywords. For more advanced use cases, the underlying logic
    can be extended or replaced, potentially leveraging NLP libraries.
    """

    _DEFAULT_STOPWORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
        "for", "with", "without", "at", "by", "on", "in", "of", "to", "from",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "me", "him", "her", "us", "them", "my", "your", "his", "hers",
        "ours", "theirs", "what", "which", "who", "whom", "whose", "i", "do", "does",
        "did", "not", "no", "yes", "can", "could", "would", "should", "will",
        "shall", "may", "might", "must", "get", "got", "go", "goes", "went",
        "had", "have", "has", "make", "made", "making", "such", "as", "about",
        "above", "across", "after", "against", "along", "among", "around", "before",
        "behind", "below", "beneath", "beside", "between", "beyond", "during", "except",
        "inside", "into", "near", "off", "on", "onto", "outside", "over", "past",
        "since", "through", "throughout", "till", "up", "upon", "versus", "via",
        "within", "without", "few", "more", "most", "other", "some", "such", "too",
        "very", "just", "only", "also", "even", "many", "much", "down", "out",
        "here", "there", "where", "when", "why", "how", "all", "any", "both", "each",
        "every", "no", "none", "nothing", "nowhere", "often", "once", "once", "one",
        "several", "some", "something", "sometimes", "somewhere", "still", "such",
        "than", "then", "there", "therefore", "these", "they", "this", "those",
        "though", "through", "thus", "to", "under", "until", "up", "upon", "us",
        "various", "very", "was", "we", "well", "were", "what", "whatever", "when",
        "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
        "whereupon", "wherever", "whether", "which", "while", "whither", "who",
        "whoever", "whole", "whom", "whomever", "whose", "why", "will", "with",
        "within", "without", "wonder", "would", "yeah", "year", "yet", "you",
        "your", "yours", "yourself", "yourselves"
    }
    _DEFAULT_MIN_KEYWORD_LENGTH: int = 3

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract a list of keywords.

        The extraction process involves:
        1. Converting text to lowercase.
        2. Tokenizing text into words, filtering out non-alphanumeric characters.
        3. Removing common stopwords (configurable via `context`).
        4. Filtering out words shorter than a minimum length (configurable via `context`).
        5. Returning unique keywords.

        Args:
            data (Any): The input data, expected to be a string of text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       or configuration for the node.
                                       Expected keys:
                                       - 'stopwords' (Set[str], optional): Custom set of stopwords.
                                       - 'min_keyword_length' (int, optional): Minimum length for a keyword.

        Returns:
            List[str]: A list of extracted keywords.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If no meaningful text or keywords can be extracted.
        """
        logger.info(f"[{self.node_name}] Starting keyword extraction process.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type: expected str, got {type(data).__name__}.")
            raise TypeError(f"Input data for KeywordExtractorNode must be a string, got {type(data).__name__}.")

        if not data.strip():
            logger.warning(f"[{self.node_name}] Input data is an empty string after stripping whitespace. Returning empty list.")
            return []

        text = data.lower()

        # Get configuration from context or use defaults
        stopwords = context.get('stopwords', self._DEFAULT_STOPWORDS)
        min_keyword_length = context.get('min_keyword_length', self._DEFAULT_MIN_KEYWORD_LENGTH)

        if not isinstance(stopwords, Set):
            logger.warning(f"[{self.node_name}] 'stopwords' in context is not a set. Using default stopwords.")
            stopwords = self._DEFAULT_STOPWORDS
        if not isinstance(min_keyword_length, int) or min_keyword_length < 1:
            logger.warning(f"[{self.node_name}] 'min_keyword_length' in context is invalid. Using default minimum length ({self._DEFAULT_MIN_KEYWORD_LENGTH}).")
            min_keyword_length = self._DEFAULT_MIN_KEYWORD_LENGTH

        # Simple tokenization: split by non-alphanumeric characters and filter out empty strings
        tokens = re.findall(r'\b[a-z]+\b', text)

        extracted_keywords: List[str] = []
        for token in tokens:
            if token not in stopwords and len(token) >= min_keyword_length:
                extracted_keywords.append(token)

        # Remove duplicates and maintain a stable order (e.g., first appearance)
        unique_keywords = list(dict.fromkeys(extracted_keywords))

        if not unique_keywords:
            logger.warning(f"[{self.node_name}] No keywords extracted from the input text after filtering.")
            # Depending on requirements, could raise ValueError here, but returning empty list is safer
            # raise ValueError("No meaningful keywords could be extracted from the provided text.")

        logger.info(f"[{self.node_name}] Successfully extracted {len(unique_keywords)} keywords.")
        return unique_keywords

