import logging
import re
from typing import Any, Dict, List, Union, Iterable, Tuple, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node can be configured to find the first match or all matches,
    and to extract specific groups from the matches.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying a regular expression pattern.

        The `context` dictionary must contain:
        - 'pattern' (str): The regular expression pattern string.

        Optional parameters in `context`:
        - 'flags' (int): Bitmask of `re` flags (e.g., re.IGNORECASE, re.MULTILINE). Defaults to 0.
        - 'match_all' (bool): If True, all non-overlapping matches are found (re.findall).
                               If False (default), only the first match is searched (re.search).
        - 'group_index' (int): Relevant only when 'match_all' is False.
                                Specifies which group to return (0 for the whole match).
                                Defaults to 0. If the group does not exist for a match, None is returned.

        Args:
            data (Any): The input data to process. Can be a string or an iterable of strings.
                        If not a string or iterable of strings, a TypeError is raised.
            context (Dict[str, Any]): A dictionary containing configuration for the node.

        Returns:
            Any: The result of the regex matching.
                 - If 'data' is a string:
                    - If 'match_all' is False: Returns a string (the matched group) or None.
                    - If 'match_all' is True: Returns a List[str] or List[Tuple[str, ...]].
                 - If 'data' is an iterable of strings: Returns a List of the above possible results for each item.

        Raises:
            ValueError: If 'pattern' is missing or invalid in the context, or if the pattern is invalid.
            TypeError: If 'pattern', 'flags', or 'group_index' are of incorrect type, or if input 'data' is
                       not a string or an iterable of strings.
        """
        pattern_str = context.get('pattern')
        flags = context.get('flags', 0)
        match_all = context.get('match_all', False)
        group_index = context.get('group_index', 0)

        if not isinstance(pattern_str, str):
            logger.error("Context missing 'pattern' or 'pattern' is not a string.")
            raise ValueError("RegexMatcherNode requires 'pattern' (str) in context.")

        if not isinstance(flags, int):
            logger.error(f"Context 'flags' must be an integer, got {type(flags).__name__}.")
            raise TypeError(f"RegexMatcherNode 'flags' must be an integer, but received {type(flags).__name__}.")

        if not isinstance(group_index, int) or group_index < 0:
            logger.error(f"Context 'group_index' must be a non-negative integer, got {group_index}.")
            raise TypeError(
                f"RegexMatcherNode 'group_index' must be a non-negative integer, but received {type(group_index).__name__}."
            )

        try:
            compiled_pattern = re.compile(pattern_str, flags)
            logger.debug(f"Successfully compiled regex pattern: '{pattern_str}' with flags: {flags}")
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern_str}': {e}")
            raise ValueError(f"Invalid regex pattern: {e}") from e

        def _apply_regex_to_single_string(text: str) -> Optional[Union[str, List[Union[str, Tuple[str, ...]]]]]:
            """Applies the compiled regex to a single string."""
            if match_all:
                result = compiled_pattern.findall(text)
                logger.debug(f"findall for '{pattern_str}' on '{text[:50]}{'...' if len(text) > 50 else ''}': {result}")
                return result
            else:
                match = compiled_pattern.search(text)
                if match:
                    try:
                        extracted = match.group(group_index)
                        logger.debug(
                            f"search group({group_index}) for '{pattern_str}' on "
                            f"'{text[:50]}{'...' if len(text) > 50 else ''}': {extracted}"
                        )
                        return extracted
                    except IndexError:
                        logger.warning(
                            f"Group index {group_index} out of range for pattern '{pattern_str}' "
                            f"on text '{text[:50]}{'...' if len(text) > 50 else ''}'. Returning None."
                        )
                        return None
                logger.debug(f"No match found for '{pattern_str}' on '{text[:50]}{'...' if len(text) > 50 else ''}'.")
                return None

        if isinstance(data, str):
            return _apply_regex_to_single_string(data)
        elif isinstance(data, Iterable):
            results = []
            for item in data:
                if isinstance(item, str):
                    results.append(_apply_regex_to_single_string(item))
                else:
                    logger.warning(
                        f"Skipping non-string item in iterable data: {type(item).__name__}. "
                        "Only string elements are processed by RegexMatcherNode; appending None."
                    )
                    results.append(None) # Append None to maintain output list length corresponding to input iterable.
            return results
        else:
            logger.error(f"Input data must be a string or an iterable of strings, got {type(data).__name__}.")
            raise TypeError(
                f"RegexMatcherNode expects input 'data' to be a string or an iterable of strings, "
                f"but received {type(data).__name__}."
            )