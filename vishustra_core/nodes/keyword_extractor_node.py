
import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract relevant keywords from input text.

    This node provides a configurable approach to identify keywords by:
    1. Normalizing the input text (converting to lowercase, removing punctuation).
    2. Tokenizing the text into individual words.
    3. Filtering out common stop words, configurable via the context.
    4. Filtering out words shorter than a specified minimum length, also configurable.
    5. Returning a list of unique, cleaned keywords.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords based on configured parameters.

        Args:
            data: The input data, expected to be a string of text from which
                  keywords will be extracted.
            context: A dictionary containing runtime context and configuration.
                     Recognized keys:
                     - 'stop_words' (Set[str] or List[str]): A collection of words
                       to be excluded from the keyword list. If not provided,
                       a sensible default set is used.
                     - 'min_word_length' (int): The minimum character length a
                       word must have to be considered a keyword. Defaults to 3.

        Returns:
            A list of unique strings, sorted alphabetically, representing the
            extracted keywords.

        Raises:
            TypeError: If the input 'data' is not a string, as this node
                       specifically operates on textual content.
        """
        logger.debug(f"[{self.node_name}] Initiating keyword extraction for input of type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to extract keywords."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.lower()

        # Retrieve stop words from context, or use a default set
        default_stop_words: Set[str] = set(context.get('stop_words', [
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
            'for', 'nor', 'on', 'at', 'to', 'from', 'in', 'of', 'with', 'by',
            'this', 'that', 'it', 'its', 'he', 'she', 'we', 'they', 'you', 'your',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ]))

        # Retrieve minimum word length from context, with validation and fallback
        min_word_length = context.get('min_word_length', 3)
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(
                f"[{self.node_name}] Invalid 'min_word_length' value in context ({min_word_length}). "
                "Falling back to default minimum length of 3."
            )
            min_word_length = 3

        # Remove punctuation and split into words. Using regex to handle various punctuation.
        # This replaces any sequence of non-alphanumeric characters with a single space.
        cleaned_text = re.sub(r'[\W_]+', ' ', text)
        words = cleaned_text.split()

        extracted_keywords: Set[str] = set()
        for word in words:
            if word and word not in default_stop_words and len(word) >= min_word_length:
                extracted_keywords.add(word)

        result = sorted(list(extracted_keywords))
        logger.debug(f"[{self.node_name}] Successfully extracted {len(result)} unique keywords.")
        return result
