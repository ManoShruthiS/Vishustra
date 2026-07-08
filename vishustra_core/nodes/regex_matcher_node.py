
import logging
import re
from typing import Any, Dict, List, Union, Pattern

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.
    It can process a single string or a list of strings, extracting
    a specified capture group or the entire match based on a pattern
    provided in the context.
    """

    def __init__(self):
        super().__init__()
        logger.debug("RegexMatcherNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcherNode"

    def _match_single_string(self, text: str, compiled_pattern: Pattern, group_index: int) -> Union[str, None]:
        """
        Helper method to apply a compiled regex pattern to a single string.

        Args:
            text (str): The string to apply the regex to.
            compiled_pattern (Pattern): The pre-compiled regular expression object.
            group_index (int): The index of the capture group to extract.

        Returns:
            Union[str, None]: The matched string or None if no match or error.
        """
        try:
            match = compiled_pattern.search(text)
            if match:
                try:
                    return match.group(group_index)
                except IndexError:
                    logger.warning(
                        f"Node '{self.node_name}': Group index {group_index} "
                        f"out of range for pattern '{compiled_pattern.pattern}' "
                        f"on text '{text[:100]}...'. Returning group 0 (full match) as a fallback."
                    )
                    # Fallback to the full match if the requested group is out of range.
                    return match.group(0)
            else:
                logger.debug(
                    f"Node '{self.node_name}': No match found for pattern "
                    f"'{compiled_pattern.pattern}' on text '{text[:100]}...'."
                )
                return None
        except Exception as e:
            logger.error(
                f"Node '{self.node_name}': An unexpected error occurred during "
                f"single string matching for pattern '{compiled_pattern.pattern}'. Error: {e}"
            )
            return None

    def process(self, data: Any, context: Dict[str, Any]) -> Union[str, List[str], None]:
        """
        Processes the input data by applying a regular expression pattern.

        The `context` dictionary must contain:
        - 'regex_pattern' (str): The regular expression pattern string to use for matching.

        The `context` dictionary can optionally contain:
        - 'regex_group' (int, optional): The index of the capture group to extract from
                                          each match. Defaults to 0 (the entire match).

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing node-specific parameters.

        Returns:
            Union[str, List[str], None]:
                - A single string if `data` was a string and a match was found.
                - A list of strings if `data` was a list of strings, containing all found matches.
                - None if `data` was a string and no match was found, or on critical error.
                - An empty list if `data` was a list of strings and no matches were found,
                  or if all items were skipped.
        """
        pattern_str = context.get('regex_pattern')
        if not pattern_str:
            logger.error(
                f"Node '{self.node_name}': 'regex_pattern' is missing from the context. "
                "Cannot perform regex matching. Returning None."
            )
            return None

        group_index = context.get('regex_group', 0)
        if not isinstance(group_index, int) or group_index < 0:
            logger.warning(
                f"Node '{self.node_name}': Invalid 'regex_group' value '{group_index}' in context. "
                "Expected a non-negative integer. Defaulting to group 0 (full match)."
            )
            group_index = 0

        try:
            compiled_pattern = re.compile(pattern_str)
        except re.error as e:
            logger.error(
                f"Node '{self.node_name}': Invalid regex pattern '{pattern_str}'. Error: {e}"
            )
            return None

        if isinstance(data, str):
            return self._match_single_string(data, compiled_pattern, group_index)
        elif isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, str):
                    match = self._match_single_string(item, compiled_pattern, group_index)
                    if match is not None:
                        results.append(match)
                else:
                    logger.warning(
                        f"Node '{self.node_name}': Skipping non-string item in list: "
                        f"type {type(item).__name__}, value '{str(item)[:50]}...'"
                    )
            return results
        else:
            logger.error(
                f"Node '{self.node_name}': Unsupported data type '{type(data).__name__}'. "
                "Expected str or List[str]. Returning None."
            )
            return None

