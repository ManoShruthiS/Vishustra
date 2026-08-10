import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node that performs regular expression matching on input data.

    This node expects the input `data` to be a string and requires a
    `regex_pattern` string in the `context`. It can also accept optional
    `regex_flags` from the `context` to modify matching behavior.

    The node uses `re.findall` to find all non-overlapping matches of the pattern
    in the data and returns them as a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing node-specific
                                       parameters. Must include:
                                       - 'regex_pattern' (str): The regular expression pattern to use.
                                       - 'regex_flags' (int, optional): Bitmask flags for regex matching
                                                                        (e.g., re.IGNORECASE). Defaults to 0.

        Returns:
            List[str]: A list of all non-overlapping matches found.
                       Returns an empty list if no matches are found, if the input
                       data is not a string, or if the pattern is invalid/missing.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected string, got %s.",
                self.node_name, type(data).__name__
            )
            return []

        regex_pattern = context.get('regex_pattern')
        if not isinstance(regex_pattern, str):
            logger.error(
                "[%s] Missing or invalid 'regex_pattern' in context. Expected string, got %s.",
                self.node_name, type(regex_pattern).__name__
            )
            return []

        regex_flags = context.get('regex_flags', 0)
        if not isinstance(regex_flags, int):
            logger.warning(
                "[%s] Invalid 'regex_flags' in context. Expected int, got %s. Defaulting to 0.",
                self.node_name, type(regex_flags).__name__
            )
            regex_flags = 0

        try:
            compiled_pattern = re.compile(regex_pattern, regex_flags)
            matches = compiled_pattern.findall(data)
            logger.debug(
                "[%s] Successfully found %d matches for pattern '%s'.",
                self.node_name, len(matches), regex_pattern
            )
            return matches
        except re.error as e:
            logger.error(
                "[%s] Invalid regex pattern '%s': %s",
                self.node_name, regex_pattern, e
            )
            return []
        except Exception as e:
            logger.exception(
                "[%s] An unexpected error occurred during regex processing: %s",
                self.node_name, e
            )
            return []
