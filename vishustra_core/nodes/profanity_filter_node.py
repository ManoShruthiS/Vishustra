import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanity words from input text.
    It can be configured with a custom profanity list and replacement character via the context.
    """

    DEFAULT_PROFANITY_LIST = [
        "badword", "cursed", "foul", "hell", "damn", "asshole", "bitch", "crap",
        # This list should be significantly expanded or loaded from a config in a real-world scenario.
    ]
    DEFAULT_REPLACEMENT_CHAR = "*"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profanity.

        If the input `data` is not a string, it logs a warning and returns the data unchanged.
        The profanity list can be overridden by `context['profanity_filter_profanity_list']`
        (expected to be a list of strings).
        The replacement character can be overridden by `context['profanity_filter_replacement_char']`
        (expected to be a non-empty string, typically a single character like '*').

        Args:
            data: The input data to be processed. Expected to be a string for filtering.
            context: A dictionary containing contextual information, potentially
                     including configuration for the profanity filter.

        Returns:
            The processed data, which is a string with profanities replaced,
            or the original data if it was not a string or an error occurred.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Received non-string data of type %s. "
                "Profanity filter only operates on strings. Returning data unchanged.",
                self.node_name, type(data).__name__
            )
            return data

        filtered_text = data
        
        # Retrieve profanity list from context, or use default
        profanity_list = context.get(
            'profanity_filter_profanity_list',
            self.DEFAULT_PROFANITY_LIST
        )
        if not isinstance(profanity_list, list) or not all(isinstance(p, str) for p in profanity_list):
            logger.warning(
                "[%s] Invalid 'profanity_filter_profanity_list' in context. "
                "Expected a list of strings. Using default list.",
                self.node_name
            )
            profanity_list = self.DEFAULT_PROFANITY_LIST

        # Retrieve replacement character from context, or use default
        replacement_char = context.get(
            'profanity_filter_replacement_char',
            self.DEFAULT_REPLACEMENT_CHAR
        )
        if not isinstance(replacement_char, str) or not replacement_char:
            logger.warning(
                "[%s] Invalid 'profanity_filter_replacement_char' in context. "
                "Expected a non-empty string. Using default '%s'.",
                self.node_name, self.DEFAULT_REPLACEMENT_CHAR
            )
            replacement_char = self.DEFAULT_REPLACEMENT_CHAR
            
        try:
            # Define a replacer function that replaces the matched word with a string
            # of `replacement_char` characters, matching the length of the original word.
            def _replacer(match):
                return replacement_char * len(match.group(0))

            for word in profanity_list:
                # Compile regex for case-insensitive whole-word matching.
                # \b ensures word boundaries, preventing partial word matches (e.g., "classic" not matching "ass").
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                filtered_text = pattern.sub(_replacer, filtered_text)

            logger.info(
                "[%s] Successfully filtered profanity from data. "
                "Original length: %d, Filtered length: %d.",
                self.node_name, len(data), len(filtered_text)
            )
            return filtered_text

        except Exception as e:
            logger.error(
                "[%s] An unexpected error occurred during profanity filtering: %s. "
                "Returning original data.",
                self.node_name, e, exc_info=True
            )
            return data