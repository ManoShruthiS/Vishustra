import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract keywords from a given text string.

    This node performs a basic, rule-based keyword extraction by:
    1. Lowercasing the input text to standardize casing.
    2. Tokenizing the text into individual words, implicitly handling common punctuation.
    3. Filtering out a predefined set of common stop words to focus on meaningful terms.
    4. Filtering out words shorter than a configurable minimum length to remove trivial tokens.
    5. Collecting unique words to avoid duplicates in the output.

    The behavior of this node can be configured via the 'context' dictionary,
    allowing customization of the stop words list and the minimum word length.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        Args:
            data (Any): The input data, which is strictly expected to be a string
                        containing the text from which keywords should be extracted.
            context (Dict[str, Any]): A dictionary containing optional configuration
                                      parameters for the keyword extraction process.
                                      Expected keys:
                                      - 'stop_words' (List[str] or Set[str], optional):
                                        A collection of words to be excluded from the
                                        extracted keywords. Defaults to a comprehensive
                                        built-in set if not provided.
                                      - 'min_word_length' (int, optional): The minimum
                                        character length a word must have to be considered
                                        a valid keyword. Defaults to 3.

        Returns:
            List[str]: A sorted list of unique extracted keywords.

        Raises:
            ValueError: If the input 'data' is not a string, as this node
                        is designed exclusively for text processing.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractorNode received data of type {type(data)}. "
                "Expected a string for text processing."
            )
            raise ValueError("KeywordExtractorNode requires string input data.")

        # Default comprehensive stop words set
        default_stop_words: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for", "nor", "so", "yet",
            "in", "on", "at", "to", "from", "of", "with", "by", "as", "it", "its", "he", "she", "they",
            "them", "their", "this", "that", "these", "those", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "not", "no", "yes", "i", "me", "my", "you", "your", "we", "our",
            "am", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "if", "then",
            "than", "else", "when", "where", "why", "how", "what", "which", "who", "whom", "whose",
            "about", "above", "across", "after", "against", "along", "among", "around", "before",
            "behind", "below", "beneath", "beside", "between", "beyond", "down", "during", "except",
            "inside", "into", "near", "off", "on", "onto", "out", "outside", "over", "past", "since",
            "through", "under", "up", "upon", "within", "without", "every", "all", "any", "some", "most",
            "many", "few", "much", "more", "less", "just", "only", "also", "even", "such", "said", "say"
        }
        
        # Load stop words from context, converting to a set for efficient lookup
        stop_words_config = context.get('stop_words')
        if isinstance(stop_words_config, (list, set)):
            stop_words = set(stop_words_config)
        else:
            stop_words = default_stop_words
            if stop_words_config is not None: # Warn if type is unexpected but not None
                logger.warning(
                    f"Unexpected type for 'stop_words' in context: {type(stop_words_config)}. "
                    "Using default stop words set."
                )
        
        # Configure minimum word length from context
        min_word_length = context.get('min_word_length', 3)
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(
                f"Invalid 'min_word_length' in context: {min_word_length}. "
                "Falling back to default of 3."
            )
            min_word_length = 3

        logger.debug(f"Starting keyword extraction for text of length {len(data)}.")

        # 1. Lowercase the entire text for case-insensitive processing
        text_lower = data.lower()

        # 2. Tokenize the text by finding sequences of word characters.
        #    This efficiently handles punctuation and separates words.
        words = re.findall(r'\b\w+\b', text_lower)

        extracted_keywords: Set[str] = set()
        for word in words:
            # 3. Filter out stop words
            if word in stop_words:
                continue
            # 4. Filter out words shorter than the minimum allowed length
            if len(word) < min_word_length:
                continue
            
            # Add valid keywords to the set to ensure uniqueness
            extracted_keywords.add(word)

        # Convert the set of unique keywords to a sorted list for consistent output
        keywords_list = sorted(list(extracted_keywords))
        logger.debug(f"Successfully extracted {len(keywords_list)} unique keywords.")
        return keywords_list