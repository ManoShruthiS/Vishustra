import re
import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node that performs regex matching on input string data.

    This node compiles a regular expression pattern upon initialization and uses it
    to find all non-overlapping occurrences within the input data, returning the
    full matched substrings.

    Configuration Parameters:
    - pattern (str): The regular expression pattern string to be matched.
    - flags (int, optional): Bitwise flags for regex compilation (e.g., re.IGNORECASE,
                             re.MULTILINE). Defaults to 0 (no flags).

    Returns:
    - Optional[List[str]]: A list of all full matched substrings found in the data.
                           Returns an empty list if no matches are found.
                           Returns None if the input `data` is not a string.
    """

    def __init__(self, pattern: str, flags: int = 0):
        """
        Initializes the RegexMatcherNode with a regex pattern and optional flags.

        Args:
            pattern (str): The regular expression pattern string.
            flags (int, optional): Regex flags (e.g., re.IGNORECASE). Defaults to 0.

        Raises:
            ValueError: If the pattern is not a non-empty string or if it's an invalid regex.
        """
        if not isinstance(pattern, str) or not pattern:
            logger.error("Attempted to initialize RegexMatcherNode with an invalid or empty pattern.")
            raise ValueError("Regex pattern must be a non-empty string.")
        
        try:
            self._compiled_pattern = re.compile(pattern, flags)
            self._pattern_str = pattern # Store original pattern for logging and introspection
            logger.debug(
                f"RegexMatcherNode initialized successfully with pattern: '{pattern}' "
                f"and flags: {flags}."
            )
        except re.error as e:
            logger.error(f"Failed to compile regex pattern '{pattern}': {e}", exc_info=True)
            raise ValueError(f"Invalid regex pattern provided: '{pattern}'") from e

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Optional[List[str]]:
        """
        Processes the input data by attempting to find all matches for the
        configured regex pattern.

        Args:
            data (Any): The input data to process. This node expects a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. Not directly utilized
                                       by this node for its core matching logic.

        Returns:
            Optional[List[str]]: A list of all full matched substrings found.
                                 Returns an empty list if `data` is a string but no
                                 matches are found. Returns `None` if `data` is not
                                 a string, indicating an incompatible input type.
        """
        if not isinstance(data, str):
            logger.warning(
                f"RegexMatcherNode received non-string data (type: {type(data).__name__}) "
                f"for pattern '{self._pattern_str}'. Expected string for matching. "
                f"Returning None."
            )
            return None

        try:
            # Using finditer and group(0) to consistently return full matched substrings
            # regardless of whether the pattern contains capturing groups.
            matches_iterator = self._compiled_pattern.finditer(data)
            results = [match.group(0) for match in matches_iterator]
            
            if results:
                logger.debug(
                    f"RegexMatcherNode found {len(results)} matches for pattern "
                    f"'{self._pattern_str}' in input data."
                )
            else:
                logger.debug(
                    f"RegexMatcherNode found no matches for pattern "
                    f"'{self._pattern_str}' in input data."
                )
            return results
        except Exception as e:
            # This catch is for highly unexpected runtime errors, as a compiled regex
            # applied to a string generally should not raise exceptions.
            logger.error(
                f"An unexpected error occurred during regex matching with pattern "
                f"'{self._pattern_str}' on data snippet: '{str(data)[:100]}...'. Error: {e}",
                exc_info=True
            )
            # In an orchestration framework, returning an empty list might be preferable
            # to propagate an error for resilience.
            return []