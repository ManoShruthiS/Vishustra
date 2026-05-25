import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set

# Assuming BaseNode is available at this path for the framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from input text data.

    This node simulates keyword extraction by tokenizing the input text,
    removing common stop words, filtering by word length, and identifying
    the most frequent non-stop words as potential keywords.

    Configuration parameters can be provided via the 'context' dictionary,
    allowing customization of `min_word_length`, `top_n_keywords`, and `stop_words`.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts keywords from the input text data based on frequency and filtering rules.

        Args:
            data: The input text as a string from which to extract keywords.
                  Expected type: `str`.
            context: A dictionary containing configuration for the extraction process:
                     - 'min_word_length' (int, optional): Minimum character length for a
                       word to be considered a keyword. Defaults to 3.
                     - 'top_n_keywords' (int, optional): The maximum number of top
                       keywords (by frequency) to return. Defaults to 5.
                     - 'stop_words' (Set[str] or List[str], optional): A collection of
                       words to ignore during extraction. If a list is provided, it will
                       be converted to a set for efficient lookup. Defaults to a
                       predefined comprehensive list of common English stop words.

        Returns:
            A list of strings, where each string is an extracted keyword.
            The list will be empty if no suitable keywords are found or if the
            input data is empty after processing.

        Raises:
            ValueError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input type for KeywordExtractorNode. Expected 'str', but received '{type(data).__name__}'.")
            raise ValueError("KeywordExtractorNode requires string input for 'data'.")

        # --- Configuration from context, with robust defaults and validation ---
        min_word_length: int = context.get("min_word_length", 3)
        top_n_keywords: int = context.get("top_n_keywords", 5)
        
        # A comprehensive default set of English stop words
        default_stop_words: Set[str] = {
            "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
            "has", "have", "he", "her", "his", "how", "i", "if", "in", "is", "it", "its",
            "of", "on", "or", "that", "the", "their", "them", "then", "there", "these",
            "they", "this", "to", "was", "we", "what", "when", "where", "which", "who",
            "will", "with", "would", "you", "your", "about", "above", "after", "again",
            "against", "all", "am", "any", "aren't", "because", "before", "below", "between",
            "both", "but", "can't", "cannot", "could", "couldn't", "did", "didn't", "do",
            "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "further",
            "had", "hadn't", "hasn't", "haven't", "having", "he'd", "he'll", "he's", "here",
            "here's", "hers", "herself", "him", "himself", "how's", "i'd", "i'll", "i'm",
            "i've", "into", "isn't", "it's", "itself", "let's", "me", "more", "most", "mustn't",
            "my", "myself", "no", "nor", "not", "off", "once", "only", "ought", "our", "ours",
            "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
            "she's", "should", "shouldn't", "so", "some", "such", "than", "that's", "theirs",
            "themselves", "they'd", "they'll", "they're", "they've", "those", "through", "too",
            "under", "until", "up", "very", "wasn't", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what's", "when's", "where's", "which", "while", "who's", "whom", "why",
            "why's", "won't", "wouldn't", "you'd", "you'll", "you're", "you've", "yourself",
            "yourselves"
        }
        
        raw_stop_words = context.get("stop_words", default_stop_words)
        stop_words: Set[str]
        
        if isinstance(raw_stop_words, (list, set)):
            stop_words = set(raw_stop_words)
            if isinstance(raw_stop_words, list):
                logger.debug("Coerced 'stop_words' from list to set for efficient lookup.")
        else:
            logger.warning(
                f"Unexpected type for 'stop_words' in context: '{type(raw_stop_words).__name__}'. "
                "Expected 'list' or 'set'. Falling back to default stop words."
            )
            stop_words = default_stop_words

        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(f"Invalid 'min_word_length' in context: {min_word_length}. Using default of 3.")
            min_word_length = 3
        
        if not isinstance(top_n_keywords, int) or top_n_keywords < 0:
            logger.warning(f"Invalid 'top_n_keywords' in context: {top_n_keywords}. Using default of 5.")
            top_n_keywords = 5
        
        logger.debug(
            f"KeywordExtractorNode processing with: min_len={min_word_length}, top_n={top_n_keywords}, "
            f"custom_stop_words={'yes' if stop_words != default_stop_words else 'no'}."
        )

        # --- Text processing pipeline ---
        # 1. Lowercase the text
        text = data.lower()
        
        # 2. Tokenize and remove punctuation.
        #    re.findall(r'\b\w+\b', text) finds all sequences of word characters.
        words = re.findall(r'\b\w+\b', text)

        # 3. Filter words based on stop words and minimum length
        filtered_words: List[str] = [
            word for word in words
            if word not in stop_words and len(word) >= min_word_length
        ]

        if not filtered_words:
            logger.info("No candidate words found after filtering. Returning an empty list of keywords.")
            return []

        # 4. Count word frequencies
        word_counts = Counter(filtered_words)

        # 5. Get the top N most common keywords
        keywords = [word for word, count in word_counts.most_common(top_n_keywords)]
        
        logger.info(f"Successfully extracted {len(keywords)} keywords: {keywords}")
        return keywords
