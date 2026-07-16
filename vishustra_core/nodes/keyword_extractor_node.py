import logging
import re
from typing import Any, Dict, List, Set, Union

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract relevant keywords from input text data.

    This node normalizes text, removes common stopwords, and filters words based
    on minimum length to identify potential keywords. It supports configuration
    of stopwords and minimum keyword length via the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract a sorted list of unique keywords.

        The input `data` can be provided as:
        - A `str`: The raw text content directly.
        - A `Dict[str, Any]`: Expected to contain the text content under the key 'text'.

        The `context` dictionary can be used to customize the extraction behavior:
        - `'stopwords'`: An optional `list` or `set` of strings to filter out. If not
                         provided, a default set of common English stopwords is used.
        - `'min_keyword_length'`: An optional `int` specifying the minimum character
                                  length for a word to be considered a keyword (default: 3).

        Args:
            data: The input text data, either a string or a dictionary containing 'text'.
            context: A dictionary for node configuration and runtime context.

        Returns:
            A sorted list of unique strings identified as keywords.

        Raises:
            ValueError: If `data` is not a string or a dictionary with a valid 'text' key,
                        or if the extracted text content is empty after validation.
            TypeError: If 'stopwords' in `context` is not a list/set, or
                       'min_keyword_length' is not a positive integer.
        """
        text_content: str = ""

        if isinstance(data, str):
            text_content = data
        elif isinstance(data, dict) and 'text' in data and isinstance(data['text'], str):
            text_content = data['text']
        else:
            error_msg = (f"Invalid input data type for '{self.node_name}'. "
                         f"Expected string or dictionary with 'text' key, got {type(data)}.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not text_content.strip():
            logger.warning("Received empty or whitespace-only text content for keyword extraction. Returning an empty list.")
            return []

        # Default stopwords for English
        default_stopwords: Set[str] = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been",
            "to", "of", "for", "on", "in", "at", "with", "from", "by", "as", "he", "she",
            "it", "they", "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
            "his", "its", "our", "their", "this", "that", "these", "those", "have", "has",
            "had", "do", "does", "did", "not", "no", "yes", "up", "down", "out", "in", "on",
            "about", "above", "across", "after", "against", "along", "among", "around",
            "before", "behind", "below", "beneath", "beside", "between", "beyond", "during",
            "except", "inside", "into", "near", "off", "onto", "outside", "over", "past",
            "since", "through", "under", "until", "upon", "when", "where", "while", "how",
            "who", "whom", "whose", "why", "which", "what", "where", "when", "then", "there",
            "here", "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
        active_stopwords: Set[str] = default_stopwords

        if 'stopwords' in context:
            if isinstance(context['stopwords'], (list, set)):
                active_stopwords = set(context['stopwords'])
                logger.debug(f"Using custom stopwords from context (first 5): {list(active_stopwords)[:5]}...")
            else:
                error_msg = f"Context key 'stopwords' must be a list or set of strings, got {type(context['stopwords'])}."
                logger.error(error_msg)
                raise TypeError(error_msg)

        min_keyword_length: int = 3
        if 'min_keyword_length' in context:
            if isinstance(context['min_keyword_length'], int) and context['min_keyword_length'] > 0:
                min_keyword_length = context['min_keyword_length']
                logger.debug(f"Using custom min_keyword_length from context: {min_keyword_length}")
            else:
                error_msg = (f"Context key 'min_keyword_length' must be a positive integer, "
                             f"got {type(context['min_keyword_length'])} or non-positive value.")
                logger.error(error_msg)
                raise TypeError(error_msg)

        logger.info(f"Initiating keyword extraction for text of length {len(text_content)}.")

        # 1. Normalize text: convert to lowercase and remove non-alphabetic characters
        # This regex keeps only lowercase letters and spaces.
        normalized_text = re.sub(r'[^a-z\s]', '', text_content.lower())
        logger.debug(f"Normalized text snippet: '{normalized_text[:100]}{'...' if len(normalized_text) > 100 else ''}'")

        # 2. Tokenize words by splitting on whitespace
        words = normalized_text.split()
        logger.debug(f"Tokenized into {len(words)} raw words.")

        # 3. Filter out stopwords and words shorter than min_keyword_length
        extracted_keywords: Set[str] = set()
        for word in words:
            if word not in active_stopwords and len(word) >= min_keyword_length:
                extracted_keywords.add(word)

        # 4. Convert the set of unique keywords to a sorted list
        sorted_keywords = sorted(list(extracted_keywords))
        logger.info(f"Successfully extracted {len(sorted_keywords)} unique keywords.")
        logger.debug(f"Extracted keywords: {sorted_keywords}")

        return sorted_keywords