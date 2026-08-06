import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node is available in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node designed to perform regular expression matching on input data.

    This node requires a 'regex_pattern' (string) to be provided in the `context` dictionary.
    Optionally, 'regex_flags' (integer, e.g., re.IGNORECASE, re.MULTILINE) can also be
    specified in the context to modify the matching behavior.

    The input `data` can be either a single string or an iterable (list, tuple) of strings.
    The node applies the compiled regular expression using `re.findall` and aggregates
    all non-overlapping matches.

    The output is a list containing the matched results. If the pattern contains
    capturing groups, the output will be a list of tuples, where each tuple contains
    the strings corresponding to the groups for a single match. If no capturing groups
    are present, the output will be a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Union[str, tuple]]:
        """
        Processes the input data by applying a specified regular expression pattern.

        Args:
            data: The input data to be processed. Expected to be a single string or
                  an iterable (list, tuple) of strings. Non-string items in iterables
                  are skipped with a warning. Other data types result in an empty list.
            context: A dictionary containing operational parameters for the node, which must include:
                     - 'regex_pattern' (str): The regular expression pattern string to be used.
                     It can optionally include:
                     - 'regex_flags' (int, optional): A bitmask of flags from the `re` module
                                                      (e.g., `re.IGNORECASE | re.MULTILINE`).
                                                      Defaults to 0 (no flags).

        Returns:
            A list of all non-overlapping matches found. Each element in the list
            can be a string (if no capturing groups in the pattern) or a tuple of
            strings (if capturing groups are present). Returns an empty list if
            no matches are found or if the input data is of an unsupported type.

        Raises:
            ValueError: If 'regex_pattern' is missing from the context or is not a string.
            re.error: If the provided 'regex_pattern' is syntactically invalid.
        """
        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str):
            logger.error(
                f"Node '{self.node_name}' requires 'regex_pattern' of type str in context. "
                f"Received: {type(regex_pattern).__name__}."
            )
            raise ValueError(f"Missing or invalid 'regex_pattern' in context for '{self.node_name}' node.")

        regex_flags = context.get("regex_flags", 0)
        if not isinstance(regex_flags, int):
            logger.warning(
                f"Invalid type for 'regex_flags' in context for '{self.node_name}' node. "
                f"Expected int, got {type(regex_flags).__name__}. Defaulting to 0 flags."
            )
            regex_flags = 0

        try:
            compiled_pattern = re.compile(regex_pattern, regex_flags)
        except re.error as e:
            logger.error(
                f"Failed to compile regex pattern '{regex_pattern}' for '{self.node_name}' node: {e}"
            )
            raise # Re-raise the original regex compilation error

        all_matches: List[Union[str, tuple]] = []

        if isinstance(data, str):
            current_matches = compiled_pattern.findall(data)
            if current_matches:
                all_matches.extend(current_matches)
        elif isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, str):
                    current_matches = compiled_pattern.findall(item)
                    if current_matches:
                        all_matches.extend(current_matches)
                else:
                    logger.warning(
                        f"Node '{self.node_name}' skipping non-string item in iterable data: "
                        f"Type '{type(item).__name__}' encountered."
                    )
        else:
            logger.warning(
                f"Node '{self.node_name}' received unexpected data type '{type(data).__name__}'. "
                "Expected str or an iterable of str. Returning an empty list."
            )
            return [] # No matches possible for unsupported data types

        return all_matches