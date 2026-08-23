import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node designed to extract keywords from a given text string.

    This node implements a simulated keyword extraction mechanism by tokenizing
    the input text, filtering out common stop words and short terms, and
    returning a curated list of unique, potentially significant terms.

    Configuration parameters can be passed via the 'context' dictionary to
    customize the extraction process:
    - 'max_keywords' (int): The maximum number of keywords to return.
      Defaults to 5 if not provided or invalid.
    - 'min_word_length' (int): The minimum length a word must have to be
      considered a keyword. Defaults to 3 if not provided or invalid.
    - 'stop_words' (List[str]): A list of custom stop words to exclude
      from the extracted keywords. If not provided, a default set of common
      English stop words is used. Custom stop words are added to the default set.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract relevant keywords.

        Args:
            data (Any): The input data, which is expected to be a string
                        containing the text from which to extract keywords.
            context (Dict[str, Any]): A dictionary containing runtime context
                                      and configuration parameters for the node.

        Returns:
            List[str]: A list of extracted keywords. Returns an empty list
                       if the input data is invalid, empty, or no keywords
                       meet the criteria.

        Raises:
            TypeError: If the input 'data' is not a string, indicating an
                       incorrect data flow or upstream node issue.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input data for {self.node_name} must be a string, but received {type(data).__name__}."
            )

        if not data.strip():
            logger.info(f"[{self.node_name}] Received empty or whitespace-only input text. Returning empty list.")
            return []

        text = data.lower()
        
        # --- Configuration Retrieval and Validation ---
        max_keywords_raw = context.get('max_keywords', 5)
        min_word_length_raw = context.get('min_word_length', 3)
        custom_stop_words_raw = context.get('stop_words', [])

        # Default common English stop words
        default_stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for",
            "nor", "so", "at", "by", "with", "from", "of", "on", "in", "to", "as",
            "it", "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
            "them", "my", "your", "his", "its", "our", "their", "this", "that",
            "these", "those", "can", "will", "would", "should", "has", "have",
            "had", "do", "does", "did", "be", "been", "being", "not", "no", "yes",
            "etc", "etc.", "what", "where", "when", "why", "how", "which", "who", "whom"
        }
        
        # Validate max_keywords
        max_keywords = 5
        if isinstance(max_keywords_raw, int) and max_keywords_raw > 0:
            max_keywords = max_keywords_raw
        else:
            logger.warning(
                f"[{self.node_name}] Invalid 'max_keywords' in context. Expected a positive integer, received '{max_keywords_raw}'. Using default of {max_keywords}."
            )
        
        # Validate min_word_length
        min_word_length = 3
        if isinstance(min_word_length_raw, int) and min_word_length_raw > 0:
            min_word_length = min_word_length_raw
        else:
            logger.warning(
                f"[{self.node_name}] Invalid 'min_word_length' in context. Expected a positive integer, received '{min_word_length_raw}'. Using default of {min_word_length}."
            )
            
        # Validate custom_stop_words
        processed_custom_stop_words = set()
        if isinstance(custom_stop_words_raw, list):
            for sw in custom_stop_words_raw:
                if isinstance(sw, str):
                    processed_custom_stop_words.add(sw.lower())
                else:
                    logger.warning(
                        f"[{self.node_name}] 'stop_words' list in context contains non-string element '{sw}'. Ignoring this entry."
                    )
        else:
            logger.warning(
                f"[{self.node_name}] Invalid 'stop_words' in context. Expected a list of strings, received '{type(custom_stop_words_raw).__name__}'. Using only default stop words."
            )
        
        # Combine default and custom stop words
        stop_words = default_stop_words.union(processed_custom_stop_words)

        # --- Keyword Extraction Logic ---
        # Tokenize the text, keeping only alphanumeric sequences
        # This regex ensures we only get "words" composed of letters and numbers
        words = re.findall(r'\b[a-z0-9]+\b', text)
        
        extracted_keywords: List[str] = []
        seen_words = set()

        for word in words:
            # Filter criteria: not a stop word, meets minimum length, and is unique
            if word not in stop_words and len(word) >= min_word_length and word not in seen_words:
                extracted_keywords.append(word)
                seen_words.add(word)
                if len(extracted_keywords) >= max_keywords:
                    logger.debug(f"[{self.node_name}] Reached max_keywords limit ({max_keywords}). Stopping extraction.")
                    break
        
        logger.info(f"[{self.node_name}] Successfully extracted {len(extracted_keywords)} keywords from the input text.")
        return extracted_keywords
