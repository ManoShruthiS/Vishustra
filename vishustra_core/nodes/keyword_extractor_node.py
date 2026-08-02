import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# Default stop words for basic English text processing.
# This set can be extended or overridden via the context.
_DEFAULT_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "else", "for", "with", "at", "by", "on",
    "in", "of", "to", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn", "about", "above", "across", "after", "afterwards", "along",
    "already", "also", "although", "always", "among", "amongst", "amount", "around",
    "as", "atop", "away", "back", "before", "beforehand", "behind", "below",
    "beside", "besides", "between", "beyond", "bottom", "brief", "cause", "co",
    "con", "could", "due", "during", "each", "eight", "either", "eleven", "elsewhere",
    "empty", "enough", "etc", "ever", "every", "everywhere", "except", "fill",
    "first", "five", "former", "formerly", "forty", "found", "four", "free",
    "front", "full", "get", "give", "go", "had", "has", "hence",
    "her", "hers", "herself", "him", "himself", "his", "how", "however",
    "hundred", "i", "ie", "if", "inasmuch", "inc", "indeed", "into", "it", "its",
    "itself", "keep", "last", "latter", "latterly", "least", "less", "many", "may",
    "me", "meanwhile", "might", "mine", "moreover", "my", "myself", "name",
    "namely", "nearly", "never", "nevertheless", "next", "nine", "nobody",
    "none", "noone", "nothing", "nowhere", "off", "often", "on", "once",
    "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "overall", "part", "per", "perhaps", "please",
    "put", "rather", "re", "really", "regarding", "same", "say", "second",
    "secondly", "see", "seem", "seemed", "seeming", "seems", "seven", "several",
    "she", "should", "show", "side", "since", "six", "sixty", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "take", "ten", "than", "that", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore",
    "therein", "thereupon", "these", "they", "third", "thirteen", "thirty",
    "this", "those", "though", "three", "through", "throughout", "thru", "thus",
    "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
    "un", "under", "until", "unto", "up", "upon", "us", "used", "using", "various",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
    "whereupon", "wherever", "whether", "which", "while", "whither", "who",
    "whoever", "whole", "whom", "whomever", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", "zero",
}


class KeywordExtractor(BaseNode):
    """
    A Vishustra node that extracts keywords from a given text string.

    This node performs a basic keyword extraction by tokenizing the input text,
    converting words to lowercase, filtering out common stop words, and removing
    words shorter than a specified minimum length. It returns a sorted list of unique
    keywords found in the text.

    Configuration can be provided via the `context` dictionary:
    - `keyword_extractor_stop_words` (Set[str]): A set of custom stop words to use.
      If not provided, a default English stop word list is used.
    - `keyword_extractor_min_word_length` (int): The minimum length a word must
      have to be considered a keyword. Defaults to 3.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Args:
            data (Any): The input text to extract keywords from. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing execution context,
                                      which may include configuration for this node.

        Returns:
            List[str]: A sorted list of unique keywords extracted from the text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractor received invalid input type. Expected 'str', got {type(data).__name__}."
            )
            raise TypeError(
                f"KeywordExtractor expects 'data' to be a string, but received {type(data).__name__}."
            )

        text = data.strip()
        if not text:
            logger.warning("KeywordExtractor received an empty string or string with only whitespace.")
            return []

        # Convert text to lowercase for consistent processing
        lower_text = text.lower()

        # Retrieve configuration from context or use defaults
        stop_words: Set[str] = context.get("keyword_extractor_stop_words", _DEFAULT_STOP_WORDS)
        min_word_length: int = context.get("keyword_extractor_min_word_length", 3)

        if not isinstance(stop_words, set):
            logger.warning(
                f"Context 'keyword_extractor_stop_words' expected Set[str], got {type(stop_words).__name__}. "
                "Using default stop words."
            )
            stop_words = _DEFAULT_STOP_WORDS
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(
                f"Context 'keyword_extractor_min_word_length' expected positive int, got {type(min_word_length).__name__}. "
                "Using default min_word_length (3)."
            )
            min_word_length = 3

        # Tokenize the text: split by non-alphabetic characters
        # re.findall(r'[a-z]+', ...) extracts sequences of alphabetic characters
        tokens = re.findall(r'[a-z]+', lower_text)

        extracted_keywords: Set[str] = set()
        for word in tokens:
            # Filter out stop words and words shorter than min_word_length
            if word not in stop_words and len(word) >= min_word_length:
                extracted_keywords.add(word)

        # Convert to a sorted list for consistent output
        result_keywords = sorted(list(extracted_keywords))

        logger.info(f"Successfully extracted {len(result_keywords)} keywords from the input text.")
        return result_keywords
