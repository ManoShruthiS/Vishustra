import re
import logging
from typing import Any, Dict, List, Optional, Union

# Assuming BaseNode is correctly imported from its specific path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression pattern to input data.
    This node can be configured to return either all non-overlapping matches found
    or just the first match, enhancing data extraction capabilities within workflows.

    The node expects the `data` input to be either a single string or a list of strings.

    Configuration parameters for initialization:
    - `pattern` (str): The regular expression pattern to be applied.
    - `flags` (int, optional): Bitmask flags for regex compilation (e.g., `re.IGNORECASE`, `re.MULTILINE`).
                               Defaults to 0 (no special flags).
    - `return_all_matches` (bool, optional): If `True`, the node will return a list of all
                                             non-overlapping matches. If `False`, it will
                                             return only the first full match string found.
                                             Defaults to `True`.
    """

    _compiled_pattern: re.Pattern
    _return_all_matches: bool

    def __init__(self, pattern: str, flags: int = 0, return_all_matches: bool = True):
        """
        Initializes the RegexMatcherNode with a specified regex pattern and behavior.

        Args:
            pattern (str): The regular expression pattern string.
            flags (int): Optional bitmask flags for the regex engine (e.g., re.IGNORECASE).
                         Defaults to 0.
            return_all_matches (bool): Determines if all matches or only the first match
                                       should be returned. Defaults to True.

        Raises:
            ValueError: If the provided pattern is empty, not a string, or an invalid regex.
        """
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Regex pattern must be a non-empty string.")

        try:
            self._compiled_pattern = re.compile(pattern, flags)
            logger.debug(
                f"Successfully compiled regex pattern: '{pattern}' with flags: {flags}"
            )
        except re.error as e:
            logger.exception(f"Invalid regex pattern provided during initialization: '{pattern}'")
            raise ValueError(f"Invalid regex pattern: {pattern}. Error: {e}") from e

        self._return_all_matches = return_all_matches

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcherNode"

    def _process_single_string(self, text: str) -> Union[List[str], Optional[str]]:
        """
        Helper method to apply the compiled regex to a single string based on the
        node's configuration (`_return_all_matches`).

        Args:
            text (str): The string to apply the regex pattern to.

        Returns:
            Union[List[str], Optional[str]]:
                If `_return_all_matches` is True, returns `List[str]` of all matches.
                If `_return_all_matches` is False, returns `Optional[str]` of the first match,
                or `None` if no match is found.
        """
        if self._return_all_matches:
            matches = self._compiled_pattern.findall(text)
            logger.debug(
                f"Found {len(matches)} matches for pattern '{self._compiled_pattern.pattern}' "
                f"in text (first 80 chars): '{text[:80]}...'"
            )
            return matches
        else:
            match = self._compiled_pattern.search(text)
            if match:
                logger.debug(
                    f"Found first match for pattern '{self._compiled_pattern.pattern}' "
                    f"in text (first 80 chars): '{text[:80]}...'"
                )
                return match.group(0)  # Return the entire matched string
            else:
                logger.debug(
                    f"No match found for pattern '{self._compiled_pattern.pattern}' "
                    f"in text (first 80 chars): '{text[:80]}...'"
                )
                return None

    def process(
        self, data: Any, context: Dict[str, Any]
    ) -> Union[List[List[str]], List[Optional[str]], List[str], Optional[str]]:
        """
        Processes the input data by applying the configured regex pattern.

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing runtime context information.
                                      This node does not directly use the context, but it's
                                      part of the `BaseNode` interface.

        Returns:
            Union[List[List[str]], List[Optional[str]], List[str], Optional[str]]:
                The output depends on the `return_all_matches` configuration and the
                type of the input `data`:
                - If `return_all_matches` is `True`:
                    - If `data` is a `str`: returns `List[str]` (all non-overlapping matches).
                    - If `data` is `List[str]`: returns `List[List[str]]` (a list where each
                                                 inner list contains matches for the corresponding
                                                 input string).
                - If `return_all_matches` is `False`:
                    - If `data` is a `str`: returns `Optional[str]` (the first full match, or `None`).
                    - If `data` is `List[str]`: returns `List[Optional[str]]` (a list where each
                                                 element is the first match for the corresponding
                                                 input string, or `None`).

                Returns an empty list or `None` if no matches are found, according to the configuration.

        Raises:
            TypeError: If `data` is not a string or a list of strings.
        """
        if data is None:
            logger.warning("Input data to RegexMatcherNode is None. Returning an empty result.")
            return [] if self._return_all_matches else None

        if isinstance(data, str):
            return self._process_single_string(data)
        elif isinstance(data, list):
            results = []
            for i, item in enumerate(data):
                if isinstance(item, str):
                    results.append(self._process_single_string(item))
                else:
                    # Log and append an appropriate placeholder for non-string items within the list
                    logger.warning(
                        f"Item at index {i} in input list is of type {type(item).__name__}, "
                        "expected string. Skipping this item and appending default empty/None result."
                    )
                    results.append([] if self._return_all_matches else None)
            return results
        else:
            logger.error(
                f"Invalid input data type: {type(data).__name__}. "
                "RegexMatcherNode expects a string or a list of strings."
            )
            raise TypeError(
                f"RegexMatcherNode received invalid data type: {type(data).__name__}. "
                "Expected `str` or `List[str]`."
            )