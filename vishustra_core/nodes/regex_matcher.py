import re
import logging
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching and extraction
    on input string data.

    This node allows defining a regex pattern and can extract either the
    first match or all matches, optionally specifying a captured group to return.
    It's designed for parsing and extracting specific pieces of information
    from larger text blocks.
    """

    def __init__(
        self,
        pattern: str,
        group_index: int = 0,
        return_all: bool = False,
        flags: int = 0
    ):
        """
        Initializes the RegexMatcherNode with a regex pattern and configuration.

        Args:
            pattern (str): The regular expression pattern string to use.
                           Must be a non-empty string.
            group_index (int, optional): The index of the captured group to return.
                                         0 means the entire match. Defaults to 0.
                                         Must be a non-negative integer.
            return_all (bool, optional): If True, all non-overlapping matches are returned
                                         as a list. If False, only the first match is returned.
                                         Defaults to False.
            flags (int, optional): Bitmask of regex flags, e.g., re.IGNORECASE, re.MULTILINE.
                                   Defaults to 0 (no flags).

        Raises:
            ValueError: If the provided pattern is invalid, empty, or if other parameters
                        are of incorrect types or values.
        """
        if not isinstance(pattern, str) or not pattern:
            logger.error("RegexMatcherNode initialization failed: 'pattern' must be a non-empty string.")
            raise ValueError("The 'pattern' argument must be a non-empty string.")
        
        if not isinstance(group_index, int) or group_index < 0:
            logger.error(f"RegexMatcherNode initialization failed: 'group_index' must be a non-negative integer, got {group_index}.")
            raise ValueError("The 'group_index' argument must be a non-negative integer.")

        if not isinstance(return_all, bool):
            logger.error(f"RegexMatcherNode initialization failed: 'return_all' must be a boolean, got {return_all}.")
            raise ValueError("The 'return_all' argument must be a boolean.")

        if not isinstance(flags, int):
            logger.error(f"RegexMatcherNode initialization failed: 'flags' must be an integer, got {flags}.")
            raise ValueError("The 'flags' argument must be an integer (bitmask of re flags).")

        try:
            self._pattern = re.compile(pattern, flags)
        except re.error as e:
            logger.error(f"Failed to compile regex pattern '{pattern}': {e}")
            raise ValueError(f"Invalid regex pattern: {pattern}. Error: {e}") from e

        self._group_index = group_index
        self._return_all = return_all
        logger.debug(
            f"Initialized RegexMatcherNode with pattern='{pattern}', group_index={group_index}, "
            f"return_all={return_all}, flags={flags}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[Optional[str], List[str]]:
        """
        Processes the input data by applying the configured regex pattern to extract
        matching substrings.

        Args:
            data (Any): The input data to search within. This is expected to be a string.
            context (Dict[str, Any]): The execution context, which may contain additional
                                       information relevant to the pipeline run (currently unused).

        Returns:
            Union[Optional[str], List[str]]:
                - If `return_all` is False: The matched string from the specified group of the
                  first found match, or None if no match is found.
                - If `return_all` is True: A list of all matched strings from the specified
                  group of all non-overlapping matches. Returns an empty list if no matches are found.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"RegexMatcherNode expects string input for data processing, but received type {type(data)}. "
                f"Data: {data!r}"
            )
            raise TypeError(
                f"RegexMatcherNode: Input data must be a string, but got {type(data).__name__}"
            )

        if self._return_all:
            results: List[str] = []
            for match in self._pattern.finditer(data):
                try:
                    group_value = match.group(self._group_index)
                    if group_value is not None:
                        results.append(group_value)
                except IndexError:
                    logger.warning(
                        f"RegexMatcherNode: Configured group_index {self._group_index} is out of "
                        f"range for a match ('{match.group(0)}'). Skipping this specific group for this match."
                    )
                    # Continue processing other matches even if one match doesn't have the requested group.
                    pass
            logger.debug(
                f"RegexMatcherNode completed processing with {len(results)} matches for pattern '{self._pattern.pattern}'."
            )
            return results
        else:
            match = self._pattern.search(data)
            if match:
                try:
                    result = match.group(self._group_index)
                    logger.debug(
                        f"RegexMatcherNode found first match '{result}' for pattern '{self._pattern.pattern}'."
                    )
                    return result
                except IndexError:
                    logger.error(
                        f"RegexMatcherNode: Configured group_index {self._group_index} "
                        f"is out of range for the first found match ('{match.group(0)}'). "
                        f"Returning None."
                    )
                    return None
            else:
                logger.debug(f"RegexMatcherNode found no match for pattern '{self._pattern.pattern}'.")
                return None