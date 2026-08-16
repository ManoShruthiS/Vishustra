import logging
import re
from typing import Any, Dict, List, Optional

# Assuming this import path exists within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out specified profanity words from input text.

    This node replaces identified profane words with a configurable mask (default: '***').
    The filtering mechanism is case-insensitive and employs word boundary matching
    to prevent unintended replacements within non-profane words.
    """

    def __init__(self, profanity_words: Optional[List[str]] = None, replacement_mask: str = "***"):
        """
        Initializes the ProfanityFilterNode with a list of profanity words and a replacement mask.

        Args:
            profanity_words: An optional list of strings that should be considered profane.
                             If `None`, a default set of common English profanities is utilized.
                             Each word will be treated as case-insensitive.
            replacement_mask: The string used to replace any identified profanity words.
                              Defaults to '***'.
        """
        # Validate and set profanity words
        if profanity_words is None:
            self._profanity_words = ["fuck", "shit", "ass", "bitch", "cunt", "damn", "cock", "pussy", "nigger"]
            logger.info("No profanity words provided; using a default list for filtering.")
        elif not isinstance(profanity_words, list) or not all(isinstance(word, str) for word in profanity_words):
            logger.error(
                f"Invalid 'profanity_words' provided during initialization. Expected a list of strings, "
                f"but received type {type(profanity_words)}. Initializing with an empty profanity list."
            )
            self._profanity_words = []
        else:
            self._profanity_words = profanity_words

        # Validate and set replacement mask
        if not isinstance(replacement_mask, str):
            logger.error(
                f"Invalid 'replacement_mask' provided. Expected a string, but received type "
                f"{type(replacement_mask)}. Falling back to default mask '***'."
            )
            self._replacement_mask = "***"
        else:
            self._replacement_mask = replacement_mask

        # Compile regex pattern for efficient and robust filtering
        if self._profanity_words:
            # Escape special regex characters in each profanity word
            escaped_words = [re.escape(word) for word in self._profanity_words]
            # Create a pattern that matches any of the words, case-insensitively, with word boundaries
            self._profanity_pattern = re.compile(
                r'\b(' + '|'.join(escaped_words) + r')\b',
                re.IGNORECASE
            )
            logger.debug(f"Profanity filter initialized with pattern: '{self._profanity_pattern.pattern}'")
        else:
            self._profanity_pattern = None
            logger.warning("ProfanityFilterNode initialized with no profanity words; no filtering will be performed.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out profanity words.

        If the input `data` is a string, it scans for known profanity and replaces
        occurrences with the configured `replacement_mask`. If `data` is not a string,
        a warning is logged, and the data is returned without modification.

        Args:
            data: The input data payload to be processed. Expected to be a string.
            context: A dictionary holding contextual information relevant to the current
                     execution flow (currently not directly used by this node, but required
                     by `BaseNode` signature).

        Returns:
            The processed data (a string with profanity filtered) or the original
            data if it was not a string, or if an error occurred during processing.
        """
        if not isinstance(data, str):
            logger.warning(
                f"ProfanityFilterNode received non-string data of type {type(data)}. "
                "Profanity filtering can only be applied to strings. Returning data unchanged."
            )
            return data

        if not self._profanity_pattern:
            logger.debug("ProfanityFilterNode has no active filtering pattern. Returning original string.")
            return data

        try:
            # Apply the compiled regex pattern to replace all detected profanity
            filtered_data = self._profanity_pattern.sub(self._replacement_mask, data)

            if filtered_data != data:
                logger.info("Profanity detected and successfully filtered from the input data.")
            else:
                logger.debug("No profanity was detected in the input data.")

            return filtered_data
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during profanity filtering: {e}. "
                "Returning original data to ensure pipeline continuity.",
                exc_info=True  # Include stack trace in log for detailed debugging
            )
            return data
