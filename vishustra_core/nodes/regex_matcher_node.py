import re
import logging
from typing import Any, Dict, List, Match

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching on input data.

    This node can match against a string directly or against a specific field
    within a dictionary. It returns a list of match objects for all non-overlapping
    matches found.

    Configuration during initialization:
    - `pattern` (str): The regular expression pattern to use.
    - `flags` (int): Optional regex flags (e.g., re.IGNORECASE, re.MULTILINE).

    Context keys for process method:
    - `target_key` (str, optional): If `data` is a dictionary, this key specifies
      which field's string value to apply the regex against. If not provided
      and `data` is a dictionary, a ValueError will be raised.

    Returns:
    - `List[re.Match]`: A list of `re.Match` objects representing all non-overlapping
      matches found. Returns an empty list if no matches are found.
    """

    def __init__(self, pattern: str, flags: int = 0):
        """
        Initializes the RegexMatcherNode with a regex pattern and optional flags.

        Args:
            pattern (str): The regular expression pattern.
            flags (int): Bitmask of regex flags (e.g., re.IGNORECASE).

        Raises:
            ValueError: If the provided regex pattern is invalid.
        """
        self._pattern_string = pattern
        self._flags = flags
        try:
            self._compiled_pattern = re.compile(pattern, flags)
            logger.info(f"RegexMatcherNode initialized with pattern: '{pattern}' and flags: {flags}")
        except re.error as e:
            logger.exception(f"Failed to compile regex pattern '{pattern}': {e}")
            raise ValueError(f"Invalid regex pattern provided: {pattern}") from e

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Match[str]]:
        """
        Processes the input data by applying the configured regex pattern.

        Args:
            data (Any): The input data, expected to be a string or a dictionary.
            context (Dict[str, Any]): A dictionary containing additional runtime
                                      information, potentially including 'target_key'.

        Returns:
            List[re.Match]: A list of `re.Match` objects found. An empty list
                            is returned if no matches are found.

        Raises:
            TypeError: If the input `data` is not a string or a dictionary,
                       or if the target field in a dictionary is not a string.
            ValueError: If `data` is a dictionary but `target_key` is missing
                        from `context` or the specified key is not found in `data`.
        """
        text_to_match: str = ""

        if isinstance(data, str):
            text_to_match = data
            logger.debug(f"Matching directly on string data provided to {self.node_name}.")
        elif isinstance(data, dict):
            target_key = context.get('target_key')
            if not target_key:
                logger.error(
                    f"Context missing 'target_key' for dictionary input in {self.node_name}. "
                    "Cannot determine which field to match against."
                )
                raise ValueError("`target_key` must be provided in context for dictionary input.")

            if target_key not in data:
                logger.error(
                    f"Target key '{target_key}' not found in input data dictionary for {self.node_name}."
                )
                raise ValueError(f"Target key '{target_key}' not found in input data.")

            field_data = data[target_key]
            if not isinstance(field_data, str):
                logger.error(
                    f"Data at target key '{target_key}' is not a string "
                    f"(type: {type(field_data)}) in {self.node_name}."
                )
                raise TypeError(f"Data at '{target_key}' must be a string, got {type(field_data)}.")
            text_to_match = field_data
            logger.debug(f"Matching on target key '{target_key}' from dictionary data in {self.node_name}.")
        else:
            logger.error(
                f"Unsupported input data type: {type(data)}. Expected str or dict for {self.node_name}."
            )
            raise TypeError(f"Unsupported data type for RegexMatcherNode: {type(data)}. Expected str or dict.")

        matches: List[Match[str]] = list(self._compiled_pattern.finditer(text_to_match))

        if matches:
            logger.info(f"Found {len(matches)} matches using pattern '{self._pattern_string}' in {self.node_name}.")
        else:
            logger.debug(f"No matches found using pattern '{self._pattern_string}' in {self.node_name}.")

        return matches