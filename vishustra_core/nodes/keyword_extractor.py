import logging
import string
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node that extracts keywords from a given text input.

    This node normalizes the input text (converts to lowercase, removes punctuation),
    splits it into words, and then filters these words based on common stop words
    and minimum length criteria to identify potential keywords.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "KeywordExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract keywords.

        Expected 'data' input: A string containing the text to be processed.
        Expected 'context' parameters:
        - 'min_keyword_length' (int, optional): Minimum length for a word to be considered a keyword. Defaults to 3.
        - 'stop_words' (Set[str], optional): A set of words to be excluded from keywords. Defaults to a small English set.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       and configuration parameters for the node.

        Returns:
            List[str]: A sorted list of unique keywords extracted from the input text.
                       Returns an empty list if input data is not a string or if no
                       keywords are found.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Invalid input data type. Expected 'str', got '%s'. Returning empty list.",
                self.node_name, type(data).__name__
            )
            return []

        text = data.lower()

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = text.translate(translator)

        words = cleaned_text.split()

        # Retrieve configuration from context or use defaults
        min_keyword_length: int = context.get('min_keyword_length', 3)
        default_stop_words: Set[str] = {
            'a', 'an', 'the', 'is', 'and', 'or', 'to', 'in', 'on', 'of',
            'for', 'with', 'from', 'by', 'at', 'it', 'its', 'he', 'she',
            'we', 'they', 'you', 'i', 'me', 'us', 'him', 'her', 'them',
            'my', 'your', 'our', 'their', 'his', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'not', 'no', 'yes', 'but', 'if', 'then', 'than', 'this', 'that',
            'these', 'those', 'as', 'can', 'will', 'would', 'should', 'could',
            'get', 'got', 'go', 'went', 'make', 'made', 'about', 'just',
            'only', 'much', 'many', 'very', 'too', 'so', 'such', 'what',
            'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'where',
            'through', 'between', 'among', 'before', 'after', 'above', 'below',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don't',
            'should'
        }
        stop_words: Set[str] = context.get('stop_words', default_stop_words)

        extracted_keywords: Set[str] = set()
        for word in words:
            if len(word) >= min_keyword_length and word not in stop_words:
                extracted_keywords.add(word)

        logger.debug(
            "[%s] Extracted %d unique keywords from input text.",
            self.node_name, len(extracted_keywords)
        )
        return sorted(list(extracted_keywords))
