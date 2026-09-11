import logging
from typing import Any, Dict, List, Union
import re

# Assuming vishustra_core.nodes.base_node exists in the project root or sys.path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanity from text data.
    It replaces detected profanity with asterisks, maintaining the original length
    of the offensive word for consistency.
    """

    # A default set of profanity words (case-insensitive for filtering).
    # This list can be extended or overridden via context.
    _DEFAULT_PROFANITY_LIST = {
        "anal", "arse", "ass", "bastard", "bitch", "bollocks", "bugger", "clit",
        "cock", "cunt", "damn", "dick", "dyke", "fag", "fuck", "goddamn", "hell",
        "jizz", "kike", "motherfucker", "nigger", "piss", "prick", "pussy",
        "shit", "slut", "snatch", "tits", "twat", "wank", "whore",
    }
    _DEFAULT_REPLACEMENT_CHAR = "*"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profanity.

        This node attempts to identify and replace profanity within string data.
        It supports filtering a single string or a list of strings. Profanity
        is replaced with asterisks, matching the length of the original word.

        The `context` dictionary can be used to customize behavior:
        - `profanity_list`: A `set` of strings containing words to filter.
          If provided, it overrides the default list.
        - `replacement_char`: A single-character string used for replacement.
          Defaults to '*'.

        Args:
            data: The input data. Expected to be a `str` or `List[str]`.
                  Other types will be returned unchanged with a warning.
            context: A dictionary containing operational context, including
                     optional 'profanity_list' and 'replacement_char'.

        Returns:
            The processed data with profanity filtered, or the original data
            if the input type is not supported or an error occurs.
        """
        current_profanity_list = self._DEFAULT_PROFANITY_LIST
        current_replacement_char = self._DEFAULT_REPLACEMENT_CHAR

        # Validate and apply profanity_list from context
        context_profanity_list = context.get('profanity_list')
        if isinstance(context_profanity_list, set) and all(isinstance(p, str) for p in context_profanity_list):
            current_profanity_list = context_profanity_list
            logger.debug("Using profanity list from context.")
        elif context_profanity_list is not None:
            logger.warning(
                "Invalid 'profanity_list' provided in context. Must be a set of strings. "
                "Using default profanity list."
            )

        # Validate and apply replacement_char from context
        context_replacement_char = context.get('replacement_char')
        if isinstance(context_replacement_char, str) and len(context_replacement_char) == 1:
            current_replacement_char = context_replacement_char
            logger.debug(f"Using replacement character '{current_replacement_char}' from context.")
        elif context_replacement_char is not None:
            logger.warning(
                "Invalid 'replacement_char' provided in context. Must be a single character string. "
                "Using default replacement character."
            )

        # Compile a regex pattern for efficient, case-insensitive profanity detection.
        # This handles words boundary and common punctuation.
        profanity_pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(word) for word in current_profanity_list) + r')\b',
            re.IGNORECASE
        )

        def _filter_single_text(text: str) -> str:
            """Helper to filter profanity in a single string."""
            if not isinstance(text, str):
                logger.debug(f"Encountered non-string item for filtering: {type(text)}. Returning as-is.")
                return text

            def replacer(match):
                original_word = match.group(0)
                # Replace with asterisks, preserving original length
                return current_replacement_char * len(original_word)

            filtered_text = profanity_pattern.sub(replacer, text)
            if filtered_text != text:
                logger.debug(f"Filtered profanity in text (original start: '{text[:20]}...')")
            return filtered_text

        try:
            if isinstance(data, str):
                return _filter_single_text(data)
            elif isinstance(data, list):
                # Apply filtering to each string in the list
                return [_filter_single_text(item) for item in data]
            else:
                logger.warning(
                    f"ProfanityFilterNode received unsupported data type: {type(data)}. "
                    "Expected str or List[str]. Returning data unchanged."
                )
                return data
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during profanity filtering: {e}",
                exc_info=True # Log full traceback for debugging
            )
            # In case of an error, it's often safer to return the original data
            # to allow downstream nodes to potentially handle or log further,
            # rather than stopping the pipeline.
            return data