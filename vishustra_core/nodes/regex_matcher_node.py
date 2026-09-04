import re
import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is available in this path from the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.
    
    This node can match a single string or a list of strings against a
    specified regular expression pattern. It supports optional flags and
    can be configured to return only the first match or all matches found.
    
    Context parameters:
    - `regex_pattern` (str): The regular expression pattern to use for matching. (Required)
    - `flags` (int, optional): Regex flags (e.g., re.IGNORECASE, re.MULTILINE). Defaults to 0.
    - `return_first_match_only` (bool, optional): If True, only the first
      occurrence of a match for each input string is returned. If False,
      all non-overlapping matches are returned. Defaults to False.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], List[List[str]]]:
        """
        Processes the input data by applying a regex pattern and extracting matches.

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing node-specific parameters.
                Must include 'regex_pattern'. Can optionally include 'flags' (int)
                and 'return_first_match_only' (bool).

        Returns:
            Union[List[str], List[List[str]]]:
                - If `data` is a string: A list of matched strings from the input.
                  (Returns an empty list if no matches are found).
                - If `data` is a list of strings: A list where each element is
                  a list of matched strings corresponding to an item in the input list.
                  Non-string items in the input list will result in an empty list
                  at their corresponding position in the output.

        Raises:
            ValueError: If 'regex_pattern' is missing from the context, is not a valid
                        string, or if the provided pattern is syntactically invalid.
            TypeError: If the input 'data' is not a string or a list of strings.
        """
        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error("Context missing required 'regex_pattern' or it is not a valid non-empty string.")
            raise ValueError("RegexMatcherNode requires a 'regex_pattern' string in the context.")

        flags = context.get("flags", 0)
        if not isinstance(flags, int):
            logger.warning(
                f"Provided 'flags' in context is not an integer ({type(flags).__name__}). "
                "Defaulting to 0 (no flags) for regex compilation."
            )
            flags = 0

        return_first_match_only = context.get("return_first_match_only", False)
        if not isinstance(return_first_match_only, bool):
            logger.warning(
                f"Provided 'return_first_match_only' in context is not a boolean "
                f"({type(return_first_match_only).__name__}). Defaulting to False."
            )
            return_first_match_only = False

        try:
            compiled_regex = re.compile(regex_pattern, flags)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{regex_pattern}' provided: {e}")
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

        if isinstance(data, str):
            return self._match_single_string(data, compiled_regex, return_first_match_only)
        elif isinstance(data, list):
            results: List[List[str]] = []
            for item in data:
                if isinstance(item, str):
                    results.append(self._match_single_string(item, compiled_regex, return_first_match_only))
                else:
                    logger.warning(
                        f"Skipping non-string item of type '{type(item).__name__}' "
                        f"in input list for RegexMatcherNode. Returning an empty list for this item."
                    )
                    results.append([])
            return results
        else:
            logger.error(
                f"RegexMatcherNode received unsupported data type: {type(data).__name__}. "
                "Expected string or list of strings for processing."
            )
            raise TypeError("Input 'data' must be a string or a list of strings.")

    def _match_single_string(self, text: str, compiled_regex: 're.Pattern[str]', return_first_match_only: bool) -> List[str]:
        """
        Helper method to apply the compiled regex to a single string.

        Args:
            text (str): The string to apply the regex to.
            compiled_regex (re.Pattern[str]): The pre-compiled regular expression object.
            return_first_match_only (bool): If True, returns only the first match found.

        Returns:
            List[str]: A list of matched strings. Will be empty if no matches are found.
        """
        if return_first_match_only:
            match = compiled_regex.search(text)
            return [match.group(0)] if match else []
        else:
            return compiled_regex.findall(text)