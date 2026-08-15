import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

# Assuming BaseNode exists at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression pattern
    to the input data and extracts all matching occurrences.
    """

    def __init__(self, pattern: str, flags: int = 0, find_all: bool = True):
        """
        Initializes the RegexMatcherNode with a regular expression pattern and flags.

        Args:
            pattern: The regular expression string to use for matching. Must be a non-empty string.
            flags: Optional flags for the regex engine (e.g., re.IGNORECASE, re.MULTILINE).
                   Defaults to 0 (no flags).
            find_all: If True, all non-overlapping matches will be returned.
                      If False, only the first match found will be returned.
                      Defaults to True.

        Raises:
            ValueError: If the provided pattern is invalid or empty.
        """
        if not isinstance(pattern, str) or not pattern:
            logger.error("Attempted to initialize RegexMatcherNode with an empty or non-string pattern.")
            raise ValueError("Regex pattern must be a non-empty string.")
        if not isinstance(flags, int):
            logger.error(f"Attempted to initialize RegexMatcherNode with non-integer flags: {type(flags)}.")
            raise TypeError("Regex flags must be an integer (e.g., from the 're' module).")
        if not isinstance(find_all, bool):
            logger.error(f"Attempted to initialize RegexMatcherNode with non-boolean find_all: {type(find_all)}.")
            raise TypeError("find_all must be a boolean.")

        self._pattern_str = pattern
        self._flags = flags
        self._find_all = find_all

        try:
            self._compiled_pattern = re.compile(self._pattern_str, self._flags)
            logger.debug(f"RegexMatcherNode initialized with pattern: '{pattern}' and flags: {flags}.")
        except re.error as e:
            logger.exception(f"Failed to compile regex pattern '{pattern}'. Error: {e}")
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Union[str, Tuple[str, ...]]]:
        """
        Applies the configured regular expression to the input data and extracts matches.

        If the regex pattern contains groups, the results will be a list of tuples,
        where each tuple contains the strings matched by the groups.
        If the regex pattern contains no groups, the results will be a list of strings,
        where each string is a full match.

        Args:
            data: The string data to search within.
            context: A dictionary containing contextual information (not directly used by this node).

        Returns:
            A list of strings or tuples representing the extracted matches.
            Returns an empty list if no matches are found or if the input data is not a string.

        Raises:
            TypeError: If the input data is not a string.
            Exception: For unexpected errors during the regex processing itself.
        """
        if not isinstance(data, str):
            logger.warning(
                f"RegexMatcherNode received non-string input of type '{type(data)}'. "
                f"Expected a string. Raising TypeError."
            )
            raise TypeError(f"RegexMatcherNode expects string input, but received {type(data)}")

        results: List[Union[str, Tuple[str, ...]]] = []

        try:
            if self._find_all:
                matches = self._compiled_pattern.findall(data)
                results.extend(matches)
                logger.debug(f"Found {len(matches)} matches for pattern '{self._pattern_str}' in data.")
            else:
                match = self._compiled_pattern.search(data)
                if match:
                    if match.groups():
                        results.append(match.groups())
                        logger.debug(f"Found first match with groups for pattern '{self._pattern_str}'.")
                    else:
                        results.append(match.group(0))
                        logger.debug(f"Found first full match for pattern '{self._pattern_str}'.")
                else:
                    logger.debug(f"No match found for pattern '{self._pattern_str}' in data (find_all=False).")

        except Exception as e:
            # Catching a broad exception here for unexpected runtime issues with the 're' module
            # that were not caught during compilation.
            logger.exception(
                f"An unexpected error occurred during regex processing for pattern '{self._pattern_str}' "
                f"on data snippet '{data[:100]}...'. Error: {e}"
            )
            raise # Re-raise after logging to signal a critical failure

        return results