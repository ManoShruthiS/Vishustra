import re
import logging
from typing import Any, Dict, List, Optional, Union, Pattern, Tuple

# Assuming this path exists for the project structure defined in Vishustra
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node that performs regex matching on input data.

    This node can be configured to use different regex strategies (findall, search,
    match, fullmatch) and to extract specific capturing groups.
    """

    def __init__(self,
                 pattern: str,
                 match_strategy: str = 'findall',
                 flags: int = 0,
                 return_group_index: Optional[Union[int, str]] = None):
        """
        Initializes the RegexMatcherNode.

        Args:
            pattern (str): The regex pattern string to compile and use.
            match_strategy (str): The strategy for matching. Must be one of
                                  'findall', 'search', 'match', 'fullmatch'.
                                  Defaults to 'findall'.
            flags (int): Regex flags (e.g., re.IGNORECASE, re.MULTILINE).
                         Defaults to 0 (no flags).
            return_group_index (Optional[Union[int, str]]): If a match is found and
                                 it has groups, specifies which group to return.
                                 Can be an integer index (0 for the entire match,
                                 1 for the first group, etc.) or a string name
                                 for named groups.
                                 If None:
                                   - For 'findall': Returns a list of full matches (if no groups
                                     in pattern) or a list of tuples of groups (if groups present).
                                   - For 'search', 'match', 'fullmatch': Returns the full match string.
                                 Defaults to None.

        Raises:
            ValueError: If the match_strategy is not recognized or the pattern is invalid.
        """
        if match_strategy not in ['findall', 'search', 'match', 'fullmatch']:
            raise ValueError(
                f"Invalid match_strategy: '{match_strategy}'. "
                "Must be one of 'findall', 'search', 'match', 'fullmatch'."
            )
        self._match_strategy = match_strategy
        self._flags = flags
        self._return_group_index = return_group_index

        try:
            self._compiled_pattern: Pattern[str] = re.compile(pattern, flags)
            logger.debug(f"RegexMatcherNode initialized with pattern: '{pattern}', "
                         f"strategy: '{match_strategy}', flags: {flags}, "
                         f"return_group_index: {return_group_index}")
        except re.error as e:
            logger.error(f"Failed to compile regex pattern '{pattern}': {e}")
            raise ValueError(f"Invalid regex pattern: '{pattern}' - {e}") from e

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying the configured regex pattern.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by this
                                       node's core logic but available.

        Returns:
            Any: The result of the regex operation, which can be:
                 - For 'findall' with `return_group_index` specified: `List[str]`
                 - For 'findall' with `return_group_index` as `None`: `List[str]` or `List[Tuple[str, ...]]`
                 - For 'search', 'match', 'fullmatch': `Optional[str]` (the extracted group or full match),
                                                        or `None` if no match.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"RegexMatcherNode received non-string input: {type(data)}. "
                         "Expected a string for regex matching.")
            raise TypeError(
                f"RegexMatcherNode requires string input, but received {type(data)}."
            )

        try:
            if self._match_strategy == 'findall':
                if self._return_group_index is not None:
                    # Use finditer for consistent group extraction with findall semantics,
                    # especially when a specific group (by index or name) is requested.
                    results = []
                    for match_obj in self._compiled_pattern.finditer(data):
                        try:
                            results.append(match_obj.group(self._return_group_index))
                        except (IndexError, KeyError):
                            # This means the requested group index/name does not exist for this match.
                            # Log a warning and skip this particular match's group extraction.
                            logger.warning(
                                f"RegexMatcherNode: Group '{self._return_group_index}' "
                                f"not found for a match in pattern '{self._compiled_pattern.pattern}'. "
                                "Skipping this match's group extraction."
                            )
                    return results
                else:
                    # Default findall behavior, returning list of strings or tuples.
                    return self._compiled_pattern.findall(data)

            else:  # 'search', 'match', 'fullmatch'
                match_obj: Optional[re.Match[str]] = None
                if self._match_strategy == 'search':
                    match_obj = self._compiled_pattern.search(data)
                elif self._match_strategy == 'match':
                    match_obj = self._compiled_pattern.match(data)
                elif self._match_strategy == 'fullmatch':
                    match_obj = self._compiled_pattern.fullmatch(data)

                if match_obj:
                    if self._return_group_index is not None:
                        try:
                            return match_obj.group(self._return_group_index)
                        except (IndexError, KeyError):
                            # If the specified group is not found, return the full match (group 0)
                            # and log a warning.
                            logger.warning(
                                f"RegexMatcherNode: Group '{self._return_group_index}' "
                                f"not found for pattern '{self._compiled_pattern.pattern}'. "
                                "Returning full match instead."
                            )
                            return match_obj.group(0)
                    else:
                        # If no specific group is requested, return the full matched string.
                        return match_obj.group(0)
                else:
                    logger.debug(f"RegexMatcherNode: No match found for pattern '{self._compiled_pattern.pattern}' "
                                 f"with strategy '{self._match_strategy}' in data.")
                    return None
        except Exception as e:
            # Catch any other unexpected errors during regex execution.
            logger.error(
                f"An unexpected error occurred during regex processing for pattern "
                f"'{self._compiled_pattern.pattern}' with strategy '{self._match_strategy}': {e}",
                exc_info=True
            )
            raise  # Re-raise the exception after logging for upstream handling.