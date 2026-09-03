import re
import logging
from typing import Any, Dict, Optional, Pattern, Union

# Assuming vishustra_core is a package at the project root level
# For local testing or specific environments, one might need to adjust sys.path or
# ensure the package is installable. In a modular framework like Vishustra,
# this import path implies the structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs a regular expression search
    on the input data.

    This node expects the input `data` to be a string. It attempts to find
    a match for a pre-configured regex pattern within the input.

    The node returns an `re.Match` object if a match is found, otherwise `None`.
    It initializes with the regex pattern and optional flags, compiling the
    pattern once for efficiency.
    """

    _compiled_pattern: Pattern[str]

    def __init__(self, pattern: str, flags: int = 0):
        """
        Initializes the RegexMatcherNode with a regex pattern and optional flags.

        Args:
            pattern: The regular expression string to be compiled and used for matching.
            flags: Optional regex flags (e.g., re.IGNORECASE, re.MULTILINE).
                   Defaults to 0 (no flags).

        Raises:
            re.error: If the provided pattern is an invalid regular expression.
        """
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Regex pattern must be a non-empty string.")

        try:
            self._compiled_pattern = re.compile(pattern, flags)
            logger.debug(f"RegexMatcherNode initialized with pattern: '{pattern}' and flags: {flags}")
        except re.error as e:
            logger.error(f"Failed to compile regex pattern '{pattern}': {e}")
            raise

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Optional[re.Match[str]]:
        """
        Processes the input data by attempting to find a match for the
        configured regular expression pattern.

        Args:
            data: The input data to search within. Expected to be a string.
            context: A dictionary containing contextual information for the node.
                     Not directly used for the matching logic in this node,
                     but available for potential future extensions or logging.

        Returns:
            An `re.Match` object if the pattern is found in the data, otherwise `None`.
            Returns `None` and logs a warning if `data` is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Regex matching requires string input. Returning None."
            )
            return None

        try:
            match = self._compiled_pattern.search(data)
            if match:
                logger.debug(
                    f"[{self.node_name}] Match found for pattern '{self._compiled_pattern.pattern}' "
                    f"in data snippet: '{data[:50]}...'"
                )
            else:
                logger.debug(
                    f"[{self.node_name}] No match found for pattern '{self._compiled_pattern.pattern}' "
                    f"in data snippet: '{data[:50]}...'"
                )
            return match
        except Exception as e:
            # Catching general exceptions here to be robust against unexpected issues
            # during the search operation, though re.search is usually stable.
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during regex search: {e}",
                exc_info=True
            )
            return None
