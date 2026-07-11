import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node that performs regex matching on input data.

    This node expects the `context` dictionary to contain a 'pattern' key
    with the regex string. Optionally, a 'flags' key (an integer representing
    re.compile flags, e.g., re.IGNORECASE | re.MULTILINE) can be provided.

    The node can process a single string or a list of strings.
    - If `data` is a string, it returns a list of all non-overlapping matches.
    - If `data` is a list of strings, it returns a list of lists, where each
      inner list contains matches for the corresponding input string.

    Raises:
        ValueError: If 'pattern' is missing in the context or if the pattern is invalid.
        TypeError: If the input data is not a string or a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], List[List[str]]]:
        """
        Processes the input data by applying a regular expression.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing the processing context.
                     Must include 'pattern' (str) and can optionally include 'flags' (int).

        Returns:
            A list of strings (if data was a string) or a list of lists of strings
            (if data was a list of strings), representing the regex matches.

        Raises:
            ValueError: If 'pattern' is missing in the context or if the pattern is invalid.
            TypeError: If the input data is not a string or a list of strings.
        """
        if 'pattern' not in context:
            logger.error("RegexMatcherNode requires a 'pattern' key in the context.")
            raise ValueError("Missing 'pattern' in context for RegexMatcherNode.")

        pattern_str = context['pattern']
        flags = context.get('flags', 0)

        try:
            regex = re.compile(pattern_str, flags)
            logger.debug(f"Compiled regex pattern: '{pattern_str}' with flags: {flags}")
        except re.error as e:
            logger.exception(f"Invalid regex pattern provided: '{pattern_str}'")
            raise ValueError(f"Invalid regex pattern: {pattern_str}. Error: {e}") from e

        if isinstance(data, str):
            matches = regex.findall(data)
            logger.debug(f"Processed single string, found {len(matches)} matches.")
            return matches
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            all_matches: List[List[str]] = []
            for item in data:
                matches = regex.findall(item)
                all_matches.append(matches)
            logger.debug(f"Processed list of strings, found matches for {len(all_matches)} items.")
            return all_matches
        else:
            logger.error(f"RegexMatcherNode received unsupported data type: {type(data)}. Expected str or list[str].")
            raise TypeError(
                f"RegexMatcherNode requires data to be a string or a list of strings, "
                f"but received {type(data)}."
            )
