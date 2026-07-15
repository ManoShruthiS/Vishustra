import re
import logging
from typing import Any, Dict, List

# Assuming BaseNode is located in vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression pattern to the input data
    and returns all non-overlapping matches.

    This node expects the `context` dictionary to contain the 'regex_pattern' key
    with a string value representing the regular expression. An optional 'regex_flags'
    key (integer, e.g., re.IGNORECASE) can also be provided.

    Input `data` is expected to be a string.
    The output is a list of strings, where each element is a match found by the regex.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern.

        Args:
            data (Any): The input data to be processed, expected to be a string.
            context (Dict[str, Any]): A dictionary containing processing parameters.
                                       Must include:
                                       - 'regex_pattern' (str): The regular expression string.
                                       Optional:
                                       - 'regex_flags' (int): Flags for the `re` module (e.g., re.IGNORECASE).
                                                              Defaults to 0 if not provided.

        Returns:
            List[str]: A list of all non-overlapping matches found in the data.
                       Returns an empty list if no matches are found or if the input data
                       is empty after type validation.

        Raises:
            ValueError: If `data` is not a string, or if 'regex_pattern' is missing,
                        not a string, or empty in the context. Also if 'regex_flags'
                        is present but not an integer.
            re.error: If the provided 'regex_pattern' is syntactically invalid.
            RuntimeError: For any other unexpected errors during processing.
        """
        if not isinstance(data, str):
            logger.error("RegexMatcherNode: Input data must be a string. Received type: %s", type(data))
            raise ValueError(f"RegexMatcherNode: Input data must be a string, got {type(data).__name__}")

        if not context:
            logger.error("RegexMatcherNode: Context dictionary is required but was empty.")
            raise ValueError("RegexMatcherNode: Context dictionary is required and must not be empty.")

        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error("RegexMatcherNode: 'regex_pattern' (str) is missing, empty, or not a string in context.")
            raise ValueError("RegexMatcherNode: 'regex_pattern' (str) must be provided in context.")

        regex_flags = context.get("regex_flags", 0)
        if not isinstance(regex_flags, int):
            logger.error("RegexMatcherNode: 'regex_flags' must be an integer. Received type: %s", type(regex_flags))
            raise ValueError(f"RegexMatcherNode: 'regex_flags' must be an integer, got {type(regex_flags).__name__}")

        try:
            compiled_regex = re.compile(regex_pattern, regex_flags)
            matches = compiled_regex.findall(data)

            if not matches:
                logger.debug("RegexMatcherNode: No matches found for pattern '%s' in data.", regex_pattern)
            else:
                logger.debug("RegexMatcherNode: Found %d matches for pattern '%s'.", len(matches), regex_pattern)
            return matches

        except re.error as e:
            logger.error("RegexMatcherNode: Invalid regex pattern '%s' provided. Error: %s", regex_pattern, e)
            raise re.error(f"RegexMatcherNode: Invalid regex pattern '{regex_pattern}'. Error: {e}") from e
        except Exception as e:
            logger.critical("RegexMatcherNode: An unexpected error occurred during processing: %s", e, exc_info=True)
            raise RuntimeError(f"RegexMatcherNode: An unexpected error occurred: {e}") from e