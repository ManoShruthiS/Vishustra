
import re
import logging
from typing import Any, Dict, List, Optional, Union, Pattern

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching on input data.

    This node provides functionality to match a predefined regex pattern against
    incoming data, supporting various matching types ('search', 'fullmatch', 'findall')
    and optional regex flags. It is designed to process either a single string
    or a list of strings, returning corresponding match results.
    """

    def __init__(self, pattern: str, match_type: str = 'search', flags: int = 0):
        """
        Initializes the RegexMatcherNode with a regex pattern and match configuration.

        Args:
            pattern (str): The regular expression pattern string to be compiled and matched.
                           Must be a non-empty string.
            match_type (str): Specifies the type of regex match operation to perform.
                              Valid values are 'search', 'fullmatch', 'findall'.
                              Defaults to 'search'.
            flags (int): Bitmask of regex flags (e.g., `re.IGNORECASE`, `re.MULTILINE`).
                         Defaults to 0 (no flags).

        Raises:
            ValueError: If the `pattern` is invalid, empty, or `match_type` is not recognized.
            TypeError: If `flags` is not an integer.
        """
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Regex 'pattern' must be a non-empty string.")
        
        valid_match_types = {'search', 'fullmatch', 'findall'}
        if match_type not in valid_match_types:
            raise ValueError(
                f"Invalid 'match_type': '{match_type}'. "
                f"Must be one of {sorted(list(valid_match_types))}."
            )
        if not isinstance(flags, int):
            raise TypeError("Regex 'flags' must be an integer (e.g., re.IGNORECASE).")

        self._pattern_str: str = pattern
        self._match_type: str = match_type
        self._flags: int = flags
        
        try:
            self._compiled_pattern: Pattern = re.compile(pattern, flags)
            logger.debug(
                f"RegexMatcherNode initialized with pattern: '{pattern}', "
                f"match_type: '{match_type}', flags: {flags}"
            )
        except re.error as e:
            logger.error(f"Failed to compile regex pattern '{pattern}': {e}")
            raise ValueError(f"Invalid regex pattern provided: {pattern}") from e

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "RegexMatcher"

    def _apply_regex(self, text: str) -> Union[Optional[re.Match], List[str]]:
        """
        Helper method to apply the compiled regex pattern to a single string
        based on the configured `match_type`.

        Args:
            text (str): The string to apply the regex to.

        Returns:
            Union[Optional[re.Match], List[str]]:
                - `re.Match` object if `match_type` is 'search' or 'fullmatch' and a match is found.
                - `None` if `match_type` is 'search' or 'fullmatch' and no match is found.
                - `List[str]` containing all non-overlapping matches if `match_type` is 'findall'.

        Raises:
            RuntimeError: If an unrecognized `match_type` is encountered (should not happen
                          due to `__init__` validation).
        """
        if self._match_type == 'search':
            return self._compiled_pattern.search(text)
        elif self._match_type == 'fullmatch':
            return self._compiled_pattern.fullmatch(text)
        elif self._match_type == 'findall':
            return self._compiled_pattern.findall(text)
        else:
            # This branch should theoretically be unreachable due to `__init__` validation.
            logger.critical(
                f"Internal error: Unrecognized match_type '{self._match_type}' "
                f"encountered in _apply_regex. This indicates a programming bug."
            )
            raise RuntimeError(f"Unrecognized regex match_type: {self._match_type}")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying the configured regex pattern.

        Args:
            data (Union[str, List[str]]): The input data to match against.
                                          Can be a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing. Not directly utilized by this node,
                                      but passed as per `BaseNode` contract.

        Returns:
            Any: The result of the regex matching operation, which varies based on
                 `data` type and `match_type`:
                - If `data` is a string and `match_type` is 'search' or 'fullmatch':
                  `re.Match` object if a match is found, otherwise `None`.
                - If `data` is a string and `match_type` is 'findall':
                  `List[str]` containing all non-overlapping matches.
                - If `data` is a list of strings, returns a `List` where each element
                  is the result of applying the regex to the respective string in the input list.

        Raises:
            TypeError: If `data` is not a string or a list of strings, or if any
                       element within an input list is not a string.
            Exception: Propagates any underlying exceptions that occur during regex
                       processing, ensuring robust error handling in the pipeline.
        """
        if isinstance(data, str):
            logger.debug(
                f"Processing single string data with RegexMatcherNode. "
                f"Pattern: '{self._pattern_str}', Match type: '{self._match_type}'"
            )
            try:
                result = self._apply_regex(data)
                logger.debug(f"RegexMatcherNode result for single string: {result}")
                return result
            except Exception as e:
                logger.exception(
                    f"Error applying regex to string data (first 50 chars: "
                    f"'{data[:50]}...'). Exception: {e}"
                )
                raise # Re-raise to signal failure in the orchestration framework

        elif isinstance(data, list):
            logger.debug(
                f"Processing list of strings data with RegexMatcherNode. "
                f"Pattern: '{self._pattern_str}', Match type: '{self._match_type}'"
            )
            results = []
            for i, item in enumerate(data):
                if isinstance(item, str):
                    try:
                        results.append(self._apply_regex(item))
                    except Exception as e:
                        logger.exception(
                            f"Error applying regex to list item at index {i} "
                            f"(first 50 chars: '{item[:50]}...'). Exception: {e}"
                        )
                        raise # Re-raise for consistent error handling in pipeline
                else:
                    error_msg = (
                        f"RegexMatcherNode received a list containing a non-string item "
                        f"at index {i} (type: {type(item)}). All list items must be strings "
                        f"for processing."
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)
            logger.debug(f"RegexMatcherNode results for list of strings: {results}")
            return results
        else:
            error_msg = (
                f"RegexMatcherNode expects input 'data' to be a string or a list of strings, "
                f"but received type: {type(data)}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

