import logging
import re
from typing import Any, Dict, List, Set

# Assume BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract keywords from textual data.

    This node processes a string input, performing a series of steps to
    identify and return significant terms. The process typically includes
    tokenization, cleaning (removing punctuation and short words), and
    filtering against a configurable list of stopwords.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input data based on defined criteria.

        The `data` input is expected to be a string of text. The `context`
        dictionary can be used to configure extraction parameters such as
        minimum keyword length and a custom list of stopwords.

        Args:
            data (Any): The input data to be processed, expected to be a string.
            context (Dict[str, Any]): A dictionary containing runtime parameters
                                      for the node. Supported keys:
                                      - `min_keyword_length` (int, optional):
                                        Minimum length for a word to be considered
                                        a keyword. Defaults to 3.
                                      - `stopwords` (List[str], optional):
                                        A list of words to exclude from the
                                        extracted keywords. Defaults to a
                                        comprehensive English stopword list.

        Returns:
            List[str]: A sorted list of unique keywords extracted from the text.
                       Returns an empty list if the input data is not a string
                       or if no significant keywords are found after filtering.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "KeywordExtractorNode received non-string data. "
                "Expected 'str' for keyword extraction, but got '%s'.",
                type(data).__name__
            )
            # Raising a TypeError is often appropriate for invalid input types
            raise TypeError(f"Input data for KeywordExtractorNode must be a string, got {type(data).__name__}")

        # --- Configuration from context ---
        min_keyword_length: int = context.get("min_keyword_length", 3)
        if not isinstance(min_keyword_length, int) or min_keyword_length < 1:
            logger.warning(
                "Invalid 'min_keyword_length' value in context: %s. "
                "Defaulting to 3.", min_keyword_length
            )
            min_keyword_length = 3
        
        # Default English stopwords (a common set)
        _default_stopwords: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "then", "else", "when", "where", "how",
            "what", "who", "whom", "this", "that", "these", "those", "for",
            "with", "without", "from", "to", "at", "by", "on", "in", "of",
            "it", "its", "i", "me", "my", "you", "your", "he", "him", "his",
            "she", "her", "hers", "we", "us", "our", "they", "them", "their",
            "not", "no", "yes", "can", "will", "would", "should", "could",
            "do", "does", "did", "have", "has", "had", "just", "get", "go",
            "say", "see", "make", "take", "come", "know", "think", "look",
            "want", "give", "use", "find", "tell", "ask", "work", "seem",
            "feel", "try", "leave", "call", "many", "much", "more", "most",
            "other", "such", "only", "own", "also", "as", "about", "above",
            "across", "after", "again", "against", "all", "almost", "alone",
            "along", "already", "although", "always", "among", "amongst",
            "amount", "any", "anyhow", "anyone", "anything", "anyway",
            "anywhere", "around", "back", "before", "behind", "below",
            "beside", "besides", "between", "beyond", "both", "brief",
            "certain", "certainly", "down", "during", "each", "etc", "ever",
            "every", "everyone", "everything", "everywhere", "except",
            "far", "few", "finally", "first", "further", "generally",
            "hardly", "hence", "here", "hereafter", "hereby", "herein",
            "hereupon", "hither", "hitherto", "however",
            "ie", "indeed", "latter", "latterly", "less", "may",
            "meanwhile", "moreover", "mostly", "namely",
            "nearly", "needs", "neither", "never", "nevertheless", "next",
            "nigh", "none", "noone", "nor", "normally", "now", "nowhere",
            "oft", "often", "ok", "okay", "once", "one", "onto",
            "otherwise", "owing", "per", "perhaps", "please", "plus",
            "quite", "rather", "re", "really", "said", "same",
            "secondly", "several", "should", "show", "side", "since",
            "so", "some", "somehow", "someone", "something", "sometime",
            "sometimes", "somewhat", "somewhere", "still",
            "than", "thence", "there", "thereafter", "thereby", "therefore",
            "therein", "thereupon", "though",
            "through", "throughout", "thru", "thus", "till", "too", "toward",
            "towards", "under", "unless", "until", "up", "upon", "upwards",
            "usually", "various", "very", "via", "viz", "vol", "vs",
            "whence", "whenever", "whereafter", "whereas",
            "whereby", "wherein", "whereupon", "wherever", "whether",
            "whither", "whoever", "whole", "whomever", "whose", "why",
            "will", "within", "without", "wonder", "would", "yet",
            "yourself", "yourselves",
        }
        
        # Merge default stopwords with any provided in context, prioritizing context
        custom_stopwords_list = context.get("stopwords", [])
        if not isinstance(custom_stopwords_list, list):
            logger.warning(
                "Invalid 'stopwords' value in context. Expected a list, got %s. "
                "Ignoring custom stopwords.", type(custom_stopwords_list).__name__
            )
            stopwords: Set[str] = _default_stopwords
        else:
            stopwords: Set[str] = _default_stopwords.union(set(word.lower() for word in custom_stopwords_list))

        # --- Text Preprocessing ---
        # Convert to lowercase to ensure case-insensitive matching
        text = data.lower()

        # Remove punctuation and tokenize into words
        # Using regex to find sequences of alphanumeric characters, effectively
        # removing most punctuation and splitting on spaces/non-alphanumeric chars.
        words = re.findall(r'\b[a-z0-9]+\b', text)

        # --- Keyword Filtering ---
        extracted_keywords: List[str] = []
        for word in words:
            # Filter out stopwords and words shorter than the minimum length
            if word not in stopwords and len(word) >= min_keyword_length:
                extracted_keywords.append(word)

        # Return unique keywords, sorted alphabetically for consistent output
        unique_keywords = sorted(list(set(extracted_keywords)))
        
        if not unique_keywords:
            logger.info(
                "No keywords extracted from the input text using current configuration "
                "(min_keyword_length=%d, stopwords_count=%d).",
                min_keyword_length, len(stopwords)
            )

        return unique_keywords
