import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out profane words from input text.
    It replaces identified profanities with a configurable placeholder text.

    This node is designed to enhance content moderation capabilities within
    LLM orchestration flows by ensuring generated or processed text adheres
    to desired civility standards.
    """

    def __init__(self, replacement_text: str = "[REDACTED]"):
        """
        Initializes the ProfanityFilterNode with a configurable replacement text
        and a predefined list of profane words.

        Args:
            replacement_text: The string used to replace identified profane words.
                              Defaults to "[REDACTED]".
        """
        self._replacement_text = replacement_text
        # A simple, extensible list of common profane words.
        # In a production system, this list would typically be loaded from a
        # dedicated configuration file, database, or a more sophisticated NLP library.
        self._profane_words = [
            r"\bass\b",        # Using regex word boundaries for accurate matching
            r"\bshit\b",
            r"\bfuck\b",
            r"\bdamn\b",
            r"\bhell\b",
            r"\bbitch\b",
            r"\bcunt\b",
            r"\bnigga\b",      # Including commonly offensive racial slurs
            r"\bnigger\b"
        ]
        # Compile regex patterns once for efficiency, enabling case-insensitive matching
        self._profanity_patterns = [
            re.compile(word, re.IGNORECASE) for word in self._profane_words
        ]
        logger.debug(
            f"ProfanityFilterNode initialized with replacement: '{self._replacement_text}' "
            f"and {len(self._profanity_patterns)} profanity patterns."
        )

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out known profanities.

        The node expects string input and replaces any matches with the
        configured `replacement_text`.

        Args:
            data: The input data, which is expected to be a string containing text
                  to be filtered.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run (not directly used by this node,
                     but part of the `BaseNode` interface).

        Returns:
            The processed string with all identified profane words replaced.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during the profanity filtering process.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        original_data = data
        filtered_data = original_data
        replacements_made = 0

        logger.debug(
            f"[{self.node_name}] Starting profanity filtering. "
            f"Input preview: '{original_data[:100]}...'" if len(original_data) > 100 else f"Input: '{original_data}'"
        )

        try:
            for pattern in self._profanity_patterns:
                # Use re.subn to get the number of substitutions made
                new_filtered_data, num_substitutions = pattern.subn(self._replacement_text, filtered_data)
                if num_substitutions > 0:
                    replacements_made += num_substitutions
                    filtered_data = new_filtered_data
                    logger.debug(
                        f"[{self.node_name}] Replaced {num_substitutions} occurrences "
                        f"for pattern '{pattern.pattern}'."
                    )
            
            if replacements_made > 0:
                logger.info(
                    f"[{self.node_name}] Profanity filtering completed. "
                    f"{replacements_made} total replacements made."
                )
            else:
                logger.info(f"[{self.node_name}] Profanity filtering completed. No profanities found.")

            return filtered_data
        except Exception as e:
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred "
                f"during profanity filtering: {e}"
            )
            logger.exception(error_msg) # Logs the exception traceback
            raise RuntimeError(error_msg) from e