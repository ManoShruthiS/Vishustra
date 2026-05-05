import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A data processing node that performs regular expression matching on input data.

    This node extracts all occurrences of a specified regex pattern from input strings.
    It can process a single string or a list of strings, returning the full matches
    or a specific capturing group.

    Configuration in context:
    - 'regex_pattern' (str, required): The regular expression pattern to match.
    - 'return_group' (int or str, optional): The index (0 for full match, 1 for first group, etc.)
      or name of the capturing group to return. Defaults to 0 (the full match).
    - 'flags' (int, optional): A bitmask of re flags (e.g., re.IGNORECASE | re.MULTILINE).
      Defaults to 0.

    Input:
    - A single string or a list of strings.

    Output:
    - If input is a single string: A list of strings, where each string is a match
      or a specific capturing group.
    - If input is a list of strings: A list of lists of strings, where each inner list
      corresponds to the matches found for the respective input string.
    - Returns an empty list or list of empty lists if no matches are found,
      or if the input data is invalid.
    """

    @property
    def node_name(self) -> str:
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying a regular expression pattern.

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): The operational context, containing configuration
                                       parameters like 'regex_pattern'.

        Returns:
            Any: A list of matched strings (or groups) or a list of lists of matches.

        Raises:
            ValueError: If 'regex_pattern' is missing or invalid in the context,
                        or if input data type is unsupported.
        """
        regex_pattern = context.get('regex_pattern')
        return_group: Union[int, str, None] = context.get('return_group', 0)
        flags: int = context.get('flags', 0)

        if not isinstance(regex_pattern, str) or not regex_pattern:
            error_msg = "RegexMatcher node requires a valid 'regex_pattern' (string) in context."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            compiled_regex = re.compile(regex_pattern, flags)
            logger.debug(f"Successfully compiled regex pattern: '{regex_pattern}' with flags: {flags}")
        except re.error as e:
            error_msg = f"Invalid regex pattern '{regex_pattern}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        if isinstance(data, str):
            return self._process_single_string(data, compiled_regex, return_group)
        elif isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, str):
                    results.append(self._process_single_string(item, compiled_regex, return_group))
                else:
                    logger.warning(
                        f"Skipping non-string item in input list: {type(item).__name__}. "
                        "Expected string or list of strings."
                    )
                    results.append([]) # Append empty list for non-string items
            return results
        else:
            error_msg = (
                f"Unsupported data type for RegexMatcher: {type(data).__name__}. "
                "Expected string or list of strings."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _process_single_string(self, text: str, compiled_regex: re.Pattern, return_group: Union[int, str, None]) -> List[str]:
        """
        Helper method to process a single string against the compiled regex.
        """
        matches: List[str] = []
        for match_obj in compiled_regex.finditer(text):
            try:
                if return_group is None: # Default to full match if explicitly None
                    matches.append(match_obj.group(0))
                else:
                    matches.append(match_obj.group(return_group))
            except IndexError:
                logger.warning(
                    f"Return group index '{return_group}' not found in match. "
                    "Skipping this match's group extraction."
                )
            except KeyError: # For named groups
                logger.warning(
                    f"Return group name '{return_group}' not found in match. "
                    "Skipping this match's group extraction."
                )
            except Exception as e:
                logger.error(f"An unexpected error occurred while extracting group '{return_group}': {e}")
        logger.debug(f"Found {len(matches)} matches for text fragment using pattern '{compiled_regex.pattern}'")
        return matches