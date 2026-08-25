import re
import logging
from typing import Any, Dict, Union, List, Optional, Match

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node uses a regex pattern provided in the context to search for matches
    within the input data. It can return either the matched strings or
    the full match objects, and can find all occurrences or just the first.

    Configuration in `context`:
    - 'pattern' (str): The regex pattern string to use. (Required)
    - 'find_all' (bool, optional): If True, finds all non-overlapping matches.
                                   Defaults to False (finds only the first match).
    - 'return_match_object' (bool, optional): If True, returns `re.Match` objects.
                                               If False, returns matched strings.
                                               Defaults to False.
    - 'flags' (int, optional): Regex flags (e.g., re.IGNORECASE, re.MULTILINE).
                               Defaults to 0.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], List[Match[str]], str, Match[str], None]:
        """
        Processes the input data by applying a regex pattern.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing node-specific configuration.
                                      Must include 'pattern' (str).

        Returns:
            Union[List[str], List[Match[str]], str, Match[str], None]:
                - If 'find_all' is True and 'return_match_object' is False: A list of matched strings.
                - If 'find_all' is True and 'return_match_object' is True: A list of `re.Match` objects.
                - If 'find_all' is False and 'return_match_object' is False: The first matched string, or None.
                - If 'find_all' is False and 'return_match_object' is True: The first `re.Match` object, or None.

                Returns an empty list if `find_all` is True and no matches are found.
                Returns None if `find_all` is False and no match is found.

        Raises:
            ValueError: If 'data' is not a string, 'pattern' is missing or not a string,
                        or if the regex pattern itself is invalid.
        """
        if not isinstance(data, str):
            logger.error("Invalid input data type for RegexMatcherNode: Expected str, got %s", type(data).__name__)
            raise ValueError(f"RegexMatcherNode requires input 'data' to be a string, but received {type(data).__name__}")

        pattern_str: Optional[str] = context.get('pattern')
        if not isinstance(pattern_str, str) or not pattern_str:
            logger.error("Missing or invalid 'pattern' in context for RegexMatcherNode. Expected non-empty str.")
            raise ValueError("RegexMatcherNode requires a non-empty string 'pattern' in context.")

        find_all: bool = context.get('find_all', False)
        return_match_object: bool = context.get('return_match_object', False)
        flags: int = context.get('flags', 0)

        logger.debug("RegexMatcherNode processing data with pattern: '%s', find_all: %s, return_match_object: %s, flags: %s",
                     pattern_str, find_all, return_match_object, flags)

        try:
            compiled_pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.error("Invalid regex pattern '%s' provided to RegexMatcherNode: %s", pattern_str, e)
            raise ValueError(f"Invalid regex pattern '{pattern_str}': {e}") from e

        if find_all:
            if return_match_object:
                matches: List[Match[str]] = list(compiled_pattern.finditer(data))
                logger.debug("RegexMatcherNode found %d match objects.", len(matches))
                return matches
            else:
                matches_strings: List[str] = compiled_pattern.findall(data)
                logger.debug("RegexMatcherNode found %d matched strings.", len(matches_strings))
                return matches_strings
        else:
            match_obj: Optional[Match[str]] = compiled_pattern.search(data)
            if return_match_object:
                logger.debug("RegexMatcherNode search result (match object): %s", match_obj)
                return match_obj
            else:
                if match_obj:
                    logger.debug("RegexMatcherNode search result (matched string): %s", match_obj.group(0))
                    return match_obj.group(0)
                else:
                    logger.debug("RegexMatcherNode found no single match.")
                    return None
