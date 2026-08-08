import re
import logging
from typing import Any, Dict, List, Union, Tuple

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression pattern to input data.

    This node is designed to extract information or validate the presence of patterns
    within strings or lists of strings. It supports compiling a regex pattern
    upon initialization and applying it efficiently during processing.
    """

    def __init__(self, pattern: str, return_boolean: bool = False, flags: int = 0):
        """
        Initializes the RegexMatcherNode with a regular expression pattern.

        Args:
            pattern: The regular expression string to be used for matching.
            return_boolean: If True, the process method returns a boolean (True if
                            any match is found, False otherwise). If False, it returns
                            a list of all non-overlapping matches found by `re.findall`.
            flags: Optional regular expression flags (e.g., re.IGNORECASE, re.MULTILINE).
                   Refer to Python's `re` module documentation for available flags.

        Raises:
            ValueError: If the provided regex pattern is invalid or an empty string.
        """
        if not isinstance(pattern, str) or not pattern:
            logger.error("RegexMatcherNode must be initialized with a non-empty string pattern.")
            raise ValueError("Pattern must be a non-empty string.")

        try:
            self._compiled_pattern = re.compile(pattern, flags)
            self._pattern_str = pattern  # Store original pattern for logging/debugging
        except re.error as e:
            logger.exception(f"Invalid regex pattern provided during initialization: '{pattern}'")
            raise ValueError(f"Invalid regex pattern: {e}") from e

        self._return_boolean = return_boolean
        logger.debug(
            f"RegexMatcherNode initialized with pattern: '{pattern}', "
            f"return_boolean: {return_boolean}, flags: {flags}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying the configured regular expression pattern.

        Args:
            data: The input data to be processed. Expected to be a string or a list of strings.
                  If `data` is a list, each element will be processed individually.
            context: A dictionary containing contextual information for the processing pipeline.
                     (This node does not directly utilize the context dictionary).

        Returns:
            Any: The result of the regex operation, which varies based on configuration:
                 - If `return_boolean` is True:
                   `bool` - True if a match is found in the string, False otherwise.
                   `List[bool]` - A list of booleans if `data` was a list of strings.
                 - If `return_boolean` is False:
                   `List[str]` or `List[Tuple[str, ...]]` - A list of all non-overlapping matches
                                                            found in the string (result of `re.findall`).
                                                            The type depends on whether the pattern
                                                            contains capturing groups.
                   `List[Union[List[str], List[Tuple[str, ...]]]]` - A list of such match lists
                                                                      if `data` was a list of strings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings, or
                       if a list contains elements that are not strings.
            Exception: Propagates unexpected errors that may occur during the regex
                       processing of individual strings.
        """
        if isinstance(data, str):
            return self._process_single_string(data)
        elif isinstance(data, list):
            results = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(
                        f"Element at index {i} in list 'data' is not a string. "
                        f"Expected type 'str', but got '{type(item).__name__}'."
                    )
                    raise TypeError(
                        f"List 'data' contains non-string elements. "
                        f"Element at index {i} is of type '{type(item).__name__}', expected 'str'."
                    )
                results.append(self._process_single_string(item))
            return results
        else:
            logger.error(
                f"Invalid input data type for RegexMatcherNode. "
                f"Expected 'str' or 'list[str]', but got '{type(data).__name__}'."
            )
            raise TypeError(
                f"Invalid input data type for RegexMatcherNode. "
                f"Expected 'str' or 'list[str]', but got '{type(data).__name__}'."
            )

    def _process_single_string(self, text: str) -> Union[bool, List[Union[str, Tuple[str, ...]]]]:
        """
        Internal method to apply the configured regex pattern to a single string.

        Args:
            text: The single string to process.

        Returns:
            Union[bool, List[Union[str, Tuple[str, ...]]]]:
                - `bool` if `_return_boolean` is True.
                - `List[str]` or `List[Tuple[str, ...]]` if `_return_boolean` is False.

        Raises:
            Exception: Catches and re-raises any unexpected exceptions during regex execution.
        """
        try:
            if self._return_boolean:
                match = self._compiled_pattern.search(text)
                return bool(match)
            else:
                matches = self._compiled_pattern.findall(text)
                return matches
        except Exception as e:
            # Catching generic Exception to log unforeseen issues during regex engine execution.
            log_text_excerpt = text[:200] + "..." if len(text) > 200 else text
            logger.exception(
                f"An unexpected error occurred while applying regex pattern "
                f"'{self._pattern_str}' to text (excerpt: '{log_text_excerpt}'): {e}"
            )
            raise  # Re-raise the exception to allow upstream error handling.