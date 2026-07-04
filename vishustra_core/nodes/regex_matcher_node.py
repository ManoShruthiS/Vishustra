import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching
    on input data. It extracts all non-overlapping occurrences based on a
    pattern provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Optional[List[Union[str, Tuple[str, ...]]]]:
        """
        Processes the input data by applying a regular expression pattern
        and extracting all non-overlapping matches.

        Expected `context` keys:
        - 'pattern' (str): The regular expression pattern to use. (Required)
        - 'flags' (int, optional): Bitmask of `re.RegexFlag` (e.g., re.IGNORECASE | re.MULTILINE).
                                   Defaults to 0 (no flags).
        - 'return_full_matches_only' (bool, optional): If True, and the pattern
                                                       contains capturing groups,
                                                       the method will return
                                                       the full matched strings
                                                       instead of tuples of groups.
                                                       Defaults to False (return groups as tuples if present).

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing parameters for processing.

        Returns:
            Optional[List[Union[str, Tuple[str, ...]]]]: A list of all non-overlapping
            matches found. Each match will be either a string (if no capturing
            groups or 'return_full_matches_only' is True) or a tuple of strings
            (if capturing groups are present). Returns None if no matches are found
            or if a critical error (e.g., invalid pattern, incorrect data type) occurs.
        """
        if not isinstance(data, str):
            logger.warning(
                f"Node '{self.node_name}': Input data is not a string. "
                f"Received type: {type(data).__name__}. Returning None."
            )
            return None

        pattern_str = context.get('pattern')
        if not isinstance(pattern_str, str) or not pattern_str:
            logger.error(
                f"Node '{self.node_name}': 'pattern' key missing or not a valid "
                f"non-empty string in context. Context received: {context}"
            )
            return None

        flags = context.get('flags', 0)
        if not isinstance(flags, int):
            logger.warning(
                f"Node '{self.node_name}': 'flags' in context is not an integer. "
                f"Received type: {type(flags).__name__}. Defaulting to 0."
            )
            flags = 0

        return_full_matches_only = context.get('return_full_matches_only', False)
        if not isinstance(return_full_matches_only, bool):
            logger.warning(
                f"Node '{self.node_name}': 'return_full_matches_only' in context is not a boolean. "
                f"Received type: {type(return_full_matches_only).__name__}. Defaulting to False."
            )
            return_full_matches_only = False

        try:
            compiled_pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.error(
                f"Node '{self.node_name}': Invalid regex pattern '{pattern_str}' "
                f"with flags {flags}: {e}"
            )
            return None

        if return_full_matches_only:
            # Use finditer to get match objects, then extract the full matched string (group 0)
            matches = [m.group(0) for m in compiled_pattern.finditer(data)]
        else:
            # Use findall which returns strings (no groups) or tuples of groups
            matches = compiled_pattern.findall(data)

        if not matches:
            logger.debug(
                f"Node '{self.node_name}': No matches found for pattern '{pattern_str}' in data."
            )
            return None

        logger.debug(
            f"Node '{self.node_name}': Found {len(matches)} matches for pattern '{pattern_str}'."
        )
        return matches
