import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node designed to perform regular expression matching
    on input data.

    This node expects the input `data` to be a string and retrieves the
    `regex_pattern` from the `context` dictionary. It uses `re.findall`
    to find all non-overlapping matches of the pattern within the data.

    Context Parameters:
    - `regex_pattern` (str): The regular expression pattern to be matched. (Required)
    - `regex_flags` (int, optional): Bitmask flags to modify regex behavior,
                                     e.g., `re.IGNORECASE`, `re.MULTILINE`.
                                     Defaults to 0 (no flags applied).
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern
        and extracting all non-overlapping matches.

        Args:
            data (Any): The input data. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing parameters for processing.
                                      Must include 'regex_pattern' (str).
                                      Can optionally include 'regex_flags' (int).

        Returns:
            List[str]: A list of strings, where each string is a match found
                       by the regex pattern. Returns an empty list if no matches
                       are found, or if input validation fails and an error
                       is handled gracefully (e.g., specific errors raised).

        Raises:
            TypeError: If `data` is not a string.
            ValueError: If 'regex_pattern' is missing from `context` or is not a string.
            re.error: If the provided 'regex_pattern' is syntactically invalid.
            RuntimeError: For any other unexpected errors during regex matching.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Please provide string data."
            )
            raise TypeError(
                f"Input 'data' must be a string for RegexMatcherNode, "
                f"received {type(data).__name__}."
            )

        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'regex_pattern' in context. "
                f"Expected a string, but received '{type(regex_pattern).__name__}'."
            )
            raise ValueError(
                "Context must contain a 'regex_pattern' string for RegexMatcherNode."
            )

        regex_flags = context.get("regex_flags", 0)
        if not isinstance(regex_flags, int):
            logger.warning(
                f"[{self.node_name}] Invalid 'regex_flags' in context. "
                f"Expected an integer, but received '{type(regex_flags).__name__}'. "
                f"Defaulting to no flags (0)."
            )
            regex_flags = 0

        try:
            matches = re.findall(regex_pattern, data, flags=regex_flags)
            logger.debug(
                f"[{self.node_name}] Successfully found {len(matches)} matches "
                f"for pattern '{regex_pattern}' (first 50 chars)."
            )
            return matches
        except re.error as e:
            logger.error(
                f"[{self.node_name}] Invalid regex pattern '{regex_pattern}'. Error: {e}"
            )
            raise re.error(f"Invalid regex pattern provided to RegexMatcherNode: {e}") from e
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during regex matching."
            )
            raise RuntimeError(
                f"An unexpected error occurred in RegexMatcherNode: {e}"
            ) from e