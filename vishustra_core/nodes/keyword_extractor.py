import logging
import re
from typing import Any, Dict, List, Set

# Assuming BaseNode is available from this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract keywords from input text data.

    This node performs a series of steps:
    1. Converts the input text to lowercase.
    2. Tokenizes the text into individual words.
    3. Filters out common stopwords based on a predefined or custom list.
    4. Filters out words shorter than a specified minimum length.
    5. Returns a sorted list of unique words identified as keywords.

    Configuration parameters for stopwords and minimum keyword length can be
    supplied via the `context` dictionary, allowing for flexible usage across
    different orchestration workflows.
    """

    DEFAULT_STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "in", "on", "at", "for", "with",
        "as", "by", "from", "it", "its", "he", "she", "they", "we", "you",
        "i", "me", "him", "her", "us", "them", "my", "your", "his", "our",
        "their", "this", "that", "these", "those", "can", "will", "would",
        "should", "has", "have", "had", "do", "does", "did", "not", "no",
        "yes", "all", "any", "some", "most", "many", "other", "such",
        "only", "very", "much", "more", "less", "about", "above", "after",
        "again", "against", "all", "along", "also", "always", "among",
        "an", "and", "another", "any", "anyone", "anything", "anywhere",
        "appear", "are", "around", "as", "ask", "at", "away", "back",
        "be", "became", "because", "become", "becomes", "been", "before",
        "began", "begin", "beginning", "behind", "being", "below",
        "beside", "besides", "best", "better", "between", "beyond", "both",
        "but", "by", "came", "can", "cannot", "come", "comes", "could",
        "did", "do", "does", "doing", "done", "down", "during", "each",
        "early", "either", "else", "enough", "etc", "even", "ever",
        "every", "everyone", "everything", "everywhere", "except", "far",
        "few", "finally", "find", "first", "for", "found", "from",
        "further", "get", "gets", "give", "given", "gives", "go", "goes",
        "going", "good", "got", "had", "has", "have", "having", "he",
        "hence", "her", "here", "hers", "herself", "him", "himself",
        "his", "how", "however", "if", "in", "indeed", "instead", "into",
        "is", "it", "its", "itself", "just", "keep", "kept", "know",
        "known", "knows", "last", "later", "least", "less", "let", "like",
        "likely", "long", "made", "make", "many", "may", "maybe", "me",
        "mean", "meanwhile", "might", "more", "most", "mostly", "much",
        "must", "my", "myself", "namely", "near", "nearly", "neither",
        "never", "nevertheless", "next", "no", "none", "nor", "not",
        "nothing", "now", "nowhere", "of", "off", "often", "on", "once",
        "one", "only", "onto", "or", "order", "other", "others", "otherwise",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "part",
        "perhaps", "place", "please", "point", "present", "put", "quite",
        "rather", "really", "regarding", "report", "result", "return",
        "said", "same", "say", "says", "second", "see", "seem", "seemed",
        "seeming", "seems", "sees", "several", "shall", "she", "should",
        "show", "showed", "shown", "shows", "side", "since", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes",
        "somewhere", "still", "such", "take", "taken", "takes", "than",
        "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "therefore", "these", "they", "thing", "things", "think",
        "thinks", "this", "those", "though", "thought", "through", "thus",
        "to", "today", "together", "told", "too", "took", "toward", "turn",
        "under", "unless", "until", "up", "upon", "us", "use", "used",
        "uses", "very", "want", "wanted", "wants", "was", "we", "well",
        "went", "were", "what", "whatever", "when", "whence", "where",
        "whereas", "wherever", "whether", "which", "while", "who", "whom",
        "whose", "why", "will", "with", "within", "without", "word",
        "words", "work", "worked", "would", "yet", "you", "your", "yours",
        "yourself", "yourselves"
    }
    DEFAULT_MIN_KEYWORD_LENGTH: int = 3

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input data, expected to be a string.

        Args:
            data (Any): The input data. Must be a string containing the text
                        from which to extract keywords.
            context (Dict[str, Any]): A dictionary containing runtime configuration.
                                      Supported keys:
                                      - 'stopwords' (Set[str], optional): A custom set of
                                        stopwords to exclude. Defaults to `DEFAULT_STOPWORDS`.
                                      - 'min_keyword_length' (int, optional): Minimum character
                                        length for a word to be considered a keyword.
                                        Defaults to `DEFAULT_MIN_KEYWORD_LENGTH`.

        Returns:
            List[str]: A sorted list of unique extracted keywords.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"KeywordExtractorNode received invalid data type. Expected 'str', "
                f"but got '{type(data).__name__}'. Unable to process."
            )
            raise TypeError(
                f"Input data for KeywordExtractorNode must be a string, "
                f"but received {type(data).__name__}."
            )

        text = data.lower()

        # Retrieve stopwords from context or fall back to default
        stopwords = context.get('stopwords', self.DEFAULT_STOPWORDS)
        if not isinstance(stopwords, set):
            logger.warning(
                f"Context 'stopwords' parameter for KeywordExtractorNode is not a set. "
                f"Received type '{type(stopwords).__name__}', falling back to default stopwords."
            )
            stopwords = self.DEFAULT_STOPWORDS
        
        # Retrieve minimum keyword length from context or fall back to default
        min_keyword_length = context.get('min_keyword_length', self.DEFAULT_MIN_KEYWORD_LENGTH)
        if not isinstance(min_keyword_length, int) or min_keyword_length < 1:
            logger.warning(
                f"Context 'min_keyword_length' for KeywordExtractorNode is invalid or not an integer. "
                f"Received '{min_keyword_length}' (type: {type(min_keyword_length).__name__}), "
                f"falling back to default length ({self.DEFAULT_MIN_KEYWORD_LENGTH})."
            )
            min_keyword_length = self.DEFAULT_MIN_KEYWORD_LENGTH

        # A robust tokenizer that handles various punctuation and multiple spaces
        words = re.findall(r'\b\w+\b', text)

        extracted_keywords = set() # Use a set for efficient uniqueness
        for word in words:
            if word not in stopwords and len(word) >= min_keyword_length:
                extracted_keywords.add(word)

        # Convert back to a sorted list for consistent and predictable output
        result = sorted(list(extracted_keywords))
        logger.debug(f"KeywordExtractorNode successfully extracted {len(result)} keywords.")
        return result
