import re
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching
    on input data.

    This node initializes with a regex pattern and optional flags. When processed,
    it expects a string input and returns a list of all non-overlapping matches
    found within that string, leveraging Python's `re` module.
    """

    def __init__(self, pattern: str, flags: int = 0):
        """
        Initializes the RegexMatcherNode with a regular expression pattern
        and optional flags.

        Args:
            pattern: The regular expression string to be compiled and used for matching.
                     Must be a non-empty string.
            flags: Optional regex flags (e.g., re.IGNORECASE, re.MULTILINE).
                   Defaults to 0 (no flags applied).

        Raises:
            ValueError: If the `pattern` is not a non-empty string or if it's an
                        invalid regular expression that cannot be compiled.
        """
        if not isinstance(pattern, str) or not pattern:
            logger.error(
                "RegexMatcherNode initialization failed: 'pattern' must be a "
                "non-empty string."
            )
            raise ValueError("The 'pattern' argument must be a non-empty string.")

        try:
            self._compiled_pattern = re.compile(pattern, flags)
            # Storing the original pattern string for logging/introspection, as the
            # compiled object itself doesn't easily expose it.
            self._pattern_str = pattern
            logger.debug(
                f"RegexMatcherNode initialized successfully with pattern: '{pattern}' "
                f"and flags: {flags}."
            )
        except re.error as e:
            logger.error(
                f"Failed to compile regex pattern '{pattern}' during initialization: {e}"
            )
            raise ValueError(
                f"Invalid regex pattern provided: '{pattern}' - {e}"
            ) from e

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying the configured regex pattern
        to find all non-overlapping matches.

        Args:
            data: The input string on which regex matching will be performed.
                  Non-string inputs will raise a TypeError.
            context: A dictionary containing contextual information for the node.
                     (This node does not directly use context for its core logic).

        Returns:
            A list of strings, where each string is a match found by the regex.
            Returns an empty list if no matches are found in the input data.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during the regex matching process.
        """
        if not isinstance(data, str):
            logger.warning(
                f"RegexMatcherNode '{self.node_name}' received non-string data "
                f"(type: {type(data).__name__}). Expected a string for matching. "
                "Raising TypeError."
            )
            raise TypeError("Input 'data' must be a string for regex matching.")

        try:
            matches = self._compiled_pattern.findall(data)
            logger.debug(
                f"RegexMatcherNode '{self.node_name}' processed input data "
                f"(length: {len(data)}). Found {len(matches)} matches for pattern "
                f"'{self._pattern_str}'."
            )
            return matches
        except Exception as e:
            # Catching general exceptions from re module's findall for robustness,
            # though most common errors should be caught during compilation.
            logger.error(
                f"An unexpected error occurred during regex processing in "
                f"RegexMatcherNode '{self.node_name}': {e}",
                exc_info=True # Include traceback for detailed debugging
            )
            raise RuntimeError(
                f"Regex matching failed unexpectedly in '{self.node_name}': {e}"
            ) from e
