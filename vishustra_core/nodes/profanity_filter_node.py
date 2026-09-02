import logging
import re
from typing import Any, Dict, List, Optional

# Assuming BaseNode is available at this path within the vishustra_core package
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out common profanities from
    text data, replacing them with asterisks to censor offensive language.

    This node uses a predefined list of profanity patterns and allows for
    custom profanities to be added during initialization. It performs
    case-insensitive matching and replaces each detected profanity with
    asterisks of the same length, ensuring the original text length is
    preserved.
    """

    def __init__(self, custom_profanities: Optional[List[str]] = None):
        """
        Initializes the ProfanityFilterNode with an optional list of
        custom profanities.

        Profanities are handled using regular expressions to match whole words
        and their common variations (e.g., 'fucking', 'shitter').

        Args:
            custom_profanities: An optional list of strings or regex patterns
                                representing additional words or phrases to filter.
                                These will be added to the default list.
        """
        # Default profanity patterns using raw strings for regex to avoid backslash issues.
        # \b ensures word boundaries, \w* catches common suffixes (e.g., -ing, -er)
        self._default_profanity_patterns: List[str] = [
            r'\bfuck\w*', r'\bshit\w*', r'\bassh?\w*', r'\bbitch\w*',
            r'\bdamn\w*', r'\bcunt\w*', r'\bhell\w*', r'\bdick\w*',
            r'\btits\w*', r'\bwhore\w*', r'\bnigga\w*', r'\bslut\w*'
        ]

        # Combine default and custom profanities, ensuring uniqueness before compiling.
        all_patterns = set(self._default_profanity_patterns)
        if custom_profanities:
            all_patterns.update(custom_profanities)

        # Compile regex patterns for efficient and case-insensitive replacement.
        self._profanity_regexes = [
            re.compile(pattern, re.IGNORECASE) for pattern in sorted(list(all_patterns))
        ]
        logger.debug(f"ProfanityFilterNode initialized with {len(self._profanity_regexes)} profanity patterns.")
        # Only log a subset or summary to avoid sensitive data in logs on init if profanities are many.
        if logger.isEnabledFor(logging.DEBUG):
            sample_patterns = [p.pattern for p in self._profanity_regexes[:5]]
            logger.debug(f"Sample profanity patterns: {sample_patterns}...")


    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanities.

        If the input `data` is a string, it will be scanned for predefined
        and custom profanities. Each detected profanity will be replaced
        with a string of asterisks ('*') of the same length as the original
        profane word. If `data` is not a string, a warning is logged, and
        the original data is returned without modification.

        Args:
            data: The input data to be processed. Expected to be a string
                  for filtering.
            context: A dictionary containing contextual information for the
                     processing pipeline. This node does not explicitly use
                     the context, but it is required by the BaseNode interface.

        Returns:
            The filtered string with profanities replaced by asterisks,
            or the original data if it was not a string or an error occurred.
        """
        if not isinstance(data, str):
            logger.warning(
                f"ProfanityFilterNode received non-string data of type '{type(data).__name__}'. "
                "Returning original data without filtering."
            )
            return data

        filtered_text = data
        try:
            for regex in self._profanity_regexes:
                # Use a lambda function to replace the matched group with asterisks
                # maintaining the original length.
                filtered_text = regex.sub(lambda m: '*' * len(m.group(0)), filtered_text)
            logger.debug(
                f"ProfanityFilterNode successfully processed text. "
                f"Original (first 50 chars): '{data[:50]}...', "
                f"Filtered (first 50 chars): '{filtered_text[:50]}...'"
            )
            return filtered_text
        except Exception as e:
            logger.error(f"ProfanityFilterNode encountered an error during processing: {e}", exc_info=True)
            # In case of an unexpected error, return the original data to prevent pipeline failure
            return data
