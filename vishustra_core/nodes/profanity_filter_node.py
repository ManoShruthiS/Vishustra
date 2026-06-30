
import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node that filters out profane language from string input.

    This node identifies predefined profane words within the input text and
    replaces them with a specified replacement character (defaulting to '*'),
    maintaining the original length of the filtered word.
    """

    # A foundational, albeit simple, list of words considered profane.
    # In a production system, this would typically be loaded from a configurable
    # external source (e.g., a database, a YAML file, or a dedicated NLP library).
    _DEFAULT_PROFANE_WORDS: List[str] = [
        "asshole", "bitch", "cunt", "damn", "fuck", "hell", "shit", "bastard",
        "pussy", "dick", "tits", "wanker", "bloody", "motherfucker"
    ]

    def __init__(self, profane_words: List[str] = None, replacement_char: str = '*'):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profane_words (List[str], optional): A custom list of words to filter.
                                                  If None, a default list is used.
            replacement_char (str, optional): The character used to replace
                                              profane words. Defaults to '*'.
        """
        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            raise ValueError("Replacement character must be a single string character.")

        self._profane_words = [word.lower() for word in (profane_words if profane_words is not None else self._DEFAULT_PROFANE_WORDS)]
        self._replacement_char = replacement_char
        
        logger.debug(f"[{self.node_name}] Initialized with {len(self._profane_words)} profane words and replacement char: '{self._replacement_char}'.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profanity from the input data.

        Expected input `data` is a string. If a non-string is provided,
        a TypeError is raised.

        Args:
            data (Any): The input data to be processed, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing pipeline.

        Returns:
            Any: The filtered string with profane words replaced by replacement characters.

        Raises:
            TypeError: If the input 'data' is not a string.
            Exception: For other unexpected errors during processing.
        """
        logger.info(f"[{self.node_name}] Starting data processing.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'. "
                "Processing aborted."
            )
            raise TypeError(
                f"ProfanityFilterNode expects string input, but received "
                f"'{type(data).__name__}'."
            )

        processed_text = data
        found_profanity = False

        try:
            for word in self._profane_words:
                # Compile a regex pattern for the word, ensuring whole word match
                # and case-insensitivity. re.escape handles special characters in words.
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)

                if pattern.search(processed_text):
                    found_profanity = True
                    # Replace found words with the replacement character repeated
                    # to match the length of the original matched word.
                    processed_text = pattern.sub(
                        lambda m: self._replacement_char * len(m.group(0)),
                        processed_text
                    )
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during profanity filtering.")
            raise

        if found_profanity:
            logger.warning(f"[{self.node_name}] Profanity detected and filtered in the input data.")
        else:
            logger.debug(f"[{self.node_name}] No profanity detected in the input data.")

        logger.info(f"[{self.node_name}] Data processing completed.")
        return processed_text

