import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is correctly installed or available in the Python path
# as part of the 'vishustra_core' package structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression to input data.

    This node extracts all non-overlapping matches from the input string based on
    the provided pattern and optional flags. It's ideal for tasks like data extraction,
    validation, or filtering based on specific textual patterns.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the programmatic name of the node.
        """
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression to find matches.

        The regex pattern and optional flags are retrieved from the `context` dictionary.

        Expected `context` keys:
        - 'pattern' (str): The regular expression pattern string to be used for matching. (Required)
        - 'flags' (int, optional): A bitmask of flags from the `re` module (e.g., `re.IGNORECASE`, `re.MULTILINE`).
                                   Defaults to 0 (no flags) if not provided.

        Args:
            data (Any): The input data to be processed. This node expects `data` to be a string.
            context (Dict[str, Any]): A dictionary containing node-specific configuration parameters,
                                      including the regex 'pattern' and optional 'flags'.

        Returns:
            List[str]: A list of all non-overlapping string matches found in the input `data`.
                       Returns an empty list (`[]`) if no matches are found.

        Raises:
            TypeError: If the `data` input is not a string.
            ValueError: If 'pattern' is missing from `context` or is not a string,
                        if 'flags' is provided but is not an integer, or if the
                        provided regex 'pattern' is syntactically invalid.
            RuntimeError: For any other unexpected errors during regex processing.
        """
        if not isinstance(data, str):
            logger.error(
                "RegexMatcherNode received non-string data. Expected a string, but got type: %s",
                type(data).__name__
            )
            raise TypeError(
                f"RegexMatcherNode requires string input for 'data', "
                f"but received type '{type(data).__name__}'."
            )

        pattern_str = context.get("pattern")
        if not isinstance(pattern_str, str):
            logger.error(
                "RegexMatcherNode: 'pattern' is missing or not a string in context. "
                "Received type: %s",
                type(pattern_str).__name__ if pattern_str is not None else 'None'
            )
            raise ValueError(
                f"RegexMatcherNode requires a string 'pattern' in context, "
                f"but received '{type(pattern_str).__name__}'."
            )

        flags = context.get("flags", 0)
        if not isinstance(flags, int):
            logger.error(
                "RegexMatcherNode: 'flags' in context must be an integer bitmask. "
                "Received type: %s",
                type(flags).__name__
            )
            raise ValueError(
                f"RegexMatcherNode requires 'flags' to be an integer (bitmask), "
                f"but received type '{type(flags).__name__}'."
            )

        try:
            # Log a snippet of the data for debugging purposes, avoiding logging large inputs.
            data_snippet = data[:200] + ('...' if len(data) > 200 else '')
            logger.debug(
                "RegexMatcherNode: Attempting to match pattern '%s' (flags: %d) against data: '%s'",
                pattern_str, flags, data_snippet
            )

            compiled_pattern = re.compile(pattern_str, flags)
            matches = compiled_pattern.findall(data)

            logger.info("RegexMatcherNode: Found %d matches.", len(matches))
            return matches
        except re.error as e:
            logger.error(
                "RegexMatcherNode failed due to an invalid regex pattern '%s': %s",
                pattern_str, e
            )
            raise ValueError(f"Invalid regex pattern provided: {e}") from e
        except Exception as e:
            logger.critical(
                "An unexpected error occurred during RegexMatcherNode processing: %s", e,
                exc_info=True # Include traceback for critical errors
            )
            raise RuntimeError(f"RegexMatcherNode encountered an unexpected error: {e}") from e