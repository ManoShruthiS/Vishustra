import re
import logging
from typing import Any, Dict, List, Optional, Union, Match, Tuple

from vishustra_core.nodes.base_node import BaseNode


logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node supports various regex operations like searching for a pattern,
    checking for a full match, or finding all occurrences. The regex pattern
    and match type are provided via the 'context' dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[Match[str], List[Union[str, Tuple[str, ...]]], None]:
        """
        Processes the input data using a specified regex pattern.

        Args:
            data: The input data, expected to be a string, on which to perform regex matching.
            context: A dictionary containing operational parameters:
                - 'pattern' (str): The regex pattern string to use. (Required)
                - 'match_type' (str): The type of regex operation to perform.
                                      Can be 'search', 'fullmatch', or 'findall'.
                                      Defaults to 'search'.
                - 'flags' (int, optional): Regex flags (e.g., re.IGNORECASE, re.MULTILINE).
                                            Defaults to 0 (no flags).

        Returns:
            - If 'match_type' is 'search' or 'fullmatch': A re.Match object if a match is found,
              otherwise None.
            - If 'match_type' is 'findall': A list of strings or tuples of strings,
              representing all non-overlapping matches.
              Returns an empty list if no matches are found.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'pattern' is missing from context, or 'match_type' is invalid.
            re.error: If the provided regex 'pattern' is invalid.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string, "
                f"but received type '{type(data).__name__}'."
            )

        pattern_str = context.get("pattern")
        if not isinstance(pattern_str, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'pattern' in context. "
                f"Expected a string, got '{type(pattern_str).__name__ if pattern_str is not None else 'None'}'."
            )
            raise ValueError(
                f"Context for '{self.node_name}' must contain a valid string 'pattern'."
            )

        match_type = context.get("match_type", "search").lower()
        flags = context.get("flags", 0)

        if not isinstance(flags, int):
            logger.warning(
                f"[{self.node_name}] 'flags' in context must be an integer (bitwise OR of re flags). "
                f"Ignoring invalid flags '{flags}' of type '{type(flags).__name__}' and using default 0."
            )
            flags = 0

        try:
            compiled_pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.error(
                f"[{self.node_name}] Invalid regex pattern '{pattern_str}': {e}"
            )
            raise re.error(f"Invalid regex pattern provided: {e}") from e

        result: Union[Match[str], List[Union[str, Tuple[str, ...]]], None] = None

        if match_type == "search":
            result = compiled_pattern.search(data)
            if result:
                logger.debug(
                    f"[{self.node_name}] Search found a match for pattern '{pattern_str}'."
                )
            else:
                logger.debug(
                    f"[{self.node_name}] Search found no match for pattern '{pattern_str}'."
                )
        elif match_type == "fullmatch":
            result = compiled_pattern.fullmatch(data)
            if result:
                logger.debug(
                    f"[{self.node_name}] Fullmatch found a match for pattern '{pattern_str}'."
                )
            else:
                logger.debug(
                    f"[{self.node_name}] Fullmatch found no match for pattern '{pattern_str}'."
                )
        elif match_type == "findall":
            result = compiled_pattern.findall(data)
            logger.debug(
                f"[{self.node_name}] Findall returned {len(result)} matches for pattern '{pattern_str}'."
            )
        else:
            logger.error(
                f"[{self.node_name}] Invalid 'match_type' '{match_type}' in context. "
                f"Expected 'search', 'fullmatch', or 'findall'."
            )
            raise ValueError(
                f"Invalid 'match_type' '{match_type}'. "
                f"Must be 'search', 'fullmatch', or 'findall'."
            )
        
        return result