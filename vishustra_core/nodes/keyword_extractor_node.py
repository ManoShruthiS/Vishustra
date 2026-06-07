import logging
from typing import Any, Dict, List, Set
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract keywords from textual data.

    This node implements a simple keyword extraction mechanism based on word
    frequency, filtering out common stop words and short words. It is suitable
    for initial text analysis or as a preliminary step for more advanced NLP tasks.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data (expected to be a string) to extract a list of keywords.

        The extraction method involves:
        1. Normalizing the input text (lowercase).
        2. Splitting the text into individual words.
        3. Filtering words based on minimum length and a configurable set of stop words.
        4. Counting the frequency of remaining words.
        5. Returning the top 'N' most frequent words as keywords.

        Args:
            data: The input text as a string from which keywords are to be extracted.
            context: A dictionary containing operational parameters for the node:
                     - 'min_word_length' (int, optional): The minimum character length a word must
                       have to be considered a keyword. Defaults to 3.
                     - 'top_n_keywords' (int, optional): The maximum number of keywords to return.
                       Defaults to 5.
                     - 'stop_words' (Set[str], optional): A custom set of words to be ignored
                       during the extraction process. If not provided, a common default set is used.

        Returns:
            A list of strings, where each string is an extracted keyword. The list
            is ordered by keyword frequency in descending order.

        Raises:
            ValueError: If the input 'data' is not a string, as this node
                        specifically operates on textual content.
        """
        logger.debug(f"[{self.node_name}] Initiating keyword extraction for input of type: {type(data)}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected 'str', but received '{type(data).__name__}'.")
            raise ValueError(f"KeywordExtractorNode requires string input, but got {type(data).__name__}.")

        text = data.lower() # Convert all text to lowercase for consistent processing

        # --- Retrieve and validate configuration from context ---
        min_word_length = context.get('min_word_length', 3)
        if not isinstance(min_word_length, int) or min_word_length < 1:
            logger.warning(
                f"[{self.node_name}] Invalid 'min_word_length' in context ({min_word_length}). "
                "Defaulting to 3 characters."
            )
            min_word_length = 3

        top_n_keywords = context.get('top_n_keywords', 5)
        if not isinstance(top_n_keywords, int) or top_n_keywords < 1:
            logger.warning(
                f"[{self.node_name}] Invalid 'top_n_keywords' in context ({top_n_keywords}). "
                "Defaulting to 5 keywords."
            )
            top_n_keywords = 5

        # Define a default set of common English stop words
        default_stop_words: Set[str] = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "he", "she", "it", "they", "them", "this", "that", "there", "here",
            "in", "on", "at", "to", "of", "for", "with", "as", "by", "from",
            "i", "you", "we", "my", "your", "our", "mine", "yours", "ours",
            "not", "no", "yes", "can", "will", "would", "should", "could",
            "have", "has", "had", "do", "does", "did", "be", "been", "being",
            "which", "who", "whom", "where", "when", "why", "how", "what",
            "about", "above", "after", "again", "all", "any", "before", "below",
            "between", "both", "each", "few", "more", "most", "other", "some",
            "such", "than", "then", "up", "down", "out", "off", "only", "own",
            "same", "so", "too", "very", "s", "t", "d", "m", "ll", "ve", "re",
            "just", "don", "shouldn", "won", "can't", "wouldn't", "couldn't",
            "isn't", "aren't", "wasn't", "weren't", "doesn't", "didn't",
            "haven't", "hasn't", "hadn't", "won't", "mustn't", "needn't",
            "mightn't", "wouldn't", "shouldn't", "couldn't", "etc", "e.g.",
            "also", "however", "therefore", "thus", "moreover", "furthermore"
        }
        stop_words = context.get('stop_words', default_stop_words)
        if not isinstance(stop_words, set):
            logger.warning(
                f"[{self.node_name}] 'stop_words' in context is not a set ({type(stop_words).__name__}). "
                "Using default stop words set."
            )
            stop_words = default_stop_words

        # --- Simulated Keyword Extraction Logic ---
        words = text.split()
        word_frequencies: Dict[str, int] = {}

        for word in words:
            # Simple cleaning: remove non-alphanumeric characters
            clean_word = ''.join(char for char in word if char.isalnum())
            if not clean_word:
                continue # Skip if word becomes empty after cleaning

            # Apply filters: minimum length and stop words
            if len(clean_word) >= min_word_length and clean_word not in stop_words:
                word_frequencies[clean_word] = word_frequencies.get(clean_word, 0) + 1

        # Sort words by frequency in descending order
        sorted_candidates = sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)

        # Extract the top N keywords
        extracted_keywords = [word for word, count in sorted_candidates[:top_n_keywords]]

        logger.debug(f"[{self.node_name}] Successfully extracted {len(extracted_keywords)} keywords.")
        return extracted_keywords