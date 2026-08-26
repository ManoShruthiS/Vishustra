import logging
from typing import Any, Dict, List, Set
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractor(BaseNode):
    """
    A processing node designed to extract keywords from a given text.

    This node simulates keyword extraction by performing basic text processing:
    tokenization, lowercasing, filtering out common stopwords, and removing
    short words based on configurable parameters.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a list of keywords.

        The `data` parameter is expected to be a string containing the text
        from which keywords should be extracted.

        The `context` dictionary can be used to pass configuration parameters:
        - 'stopwords': An iterable of strings to be considered as stopwords.
                       Defaults to a common English set if not provided.
        - 'min_word_length': An integer specifying the minimum length a word
                             must have to be considered a keyword. Defaults to 3.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary for configuration and contextual data.

        Returns:
            List[str]: A sorted list of unique extracted keywords. Returns an
                       empty list if the input is invalid or no keywords are found.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"Invalid input type for KeywordExtractor. Expected 'str', but received '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.strip()
        if not text:
            logger.info("Input text is empty after stripping in KeywordExtractor. Returning an empty list.")
            return []

        # --- Configuration from context ---
        # Default common English stopwords. A more robust solution might load from a file or NLTK.
        default_stopwords: Set[str] = set(context.get('stopwords', [
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'for', 'nor', 'so', 'yet',
            'in', 'on', 'at', 'by', 'with', 'from', 'of', 'to', 'for', 'over', 'under', 'through', 'into',
            'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'we', 'they', 'i', 'me', 'you',
            'my', 'your', 'his', 'her', 'our', 'their', 'whom', 'whose', 'which', 'what', 'where', 'when',
            'why', 'how', 'do', 'don\'t', 'does', 'doesn\'t', 'did', 'didn\'t', 'can', 'can\'t', 'will',
            'won\'t', 'shall', 'should', 'could', 'would', 'may', 'might', 'must', 'been', 'being', 'be',
            'am', 'as', 'about', 'then', 'than', 'more', 'less', 'most', 'least', 'many', 'much', 'few',
            'some', 'any', 'all', 'every', 'no', 'not', 'only', 'just', 'too', 'very', 's', 't', 'm', 're',
            've', 'll', 'd', 'o', 'up', 'down', 'out', 'off', 'on', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now', 'said'
        ]))
        min_word_length: int = context.get('min_word_length', 3)

        if not isinstance(min_word_length, int) or min_word_length < 0:
            logger.warning(
                f"Invalid 'min_word_length' in context: {min_word_length}. "
                "Using default value of 3."
            )
            min_word_length = 3
        
        if not isinstance(default_stopwords, set) and not isinstance(default_stopwords, list):
            logger.warning(
                f"Invalid 'stopwords' type in context: {type(default_stopwords).__name__}. "
                "Using default set of stopwords."
            )
            default_stopwords = set() # Reset to empty to avoid error if input was bad type

        # Basic tokenization, lowercasing, and filtering for alphabetic words
        words = [
            word.lower()
            for word in text.split()
            if word.isalpha() # Ensures we only process actual words, not punctuation or numbers
        ]

        # Filter out stopwords and words shorter than the minimum length
        potential_keywords = {
            word for word in words
            if word not in default_stopwords and len(word) >= min_word_length
        }

        # Sort the keywords for consistent output
        extracted_keywords = sorted(list(potential_keywords))

        if not extracted_keywords:
            logger.debug("No keywords extracted from the text after applying filters.")
        else:
            logger.debug(f"Successfully extracted {len(extracted_keywords)} keywords.")

        return extracted_keywords