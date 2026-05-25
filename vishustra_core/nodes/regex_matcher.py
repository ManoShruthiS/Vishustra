import re
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class RegexMatcher(BaseNode):
    """
    A Vishustra node that performs regex matching on input data.

    This node expects a string `data` as input and a `regex_pattern`
    in the `context` dictionary to find all non-overlapping matches.
    It returns a list of all matched substrings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern.

        Args:
            data: The input string data to be matched against.
                  Expected to be a string.
            context: A dictionary containing operational parameters.
                     Must include 'regex_pattern' (str) which is the regular
                     expression to use.
                     Optionally, can include 'flags' (int) for regex flags
                     like `re.IGNORECASE` or `re.MULTILINE`.

        Returns:
            A list of all non-overlapping matched substrings.
            Returns an empty list if no matches are found.

        Raises:
            ValueError: If 'data' is not a string, if 'regex_pattern' is missing
                        from context, if 'regex_pattern' is not a string, or
                        if the provided regex pattern is invalid.
            RuntimeError: For any unexpected errors during the matching process.
        """
        if not isinstance(data, str):
            logger.error(
                "RegexMatcher node received non-string data. Expected 'str', got '%s'.",
                type(data).__name__,
            )
            raise ValueError(
                f"Invalid input data type for RegexMatcher. Expected str, got {type(data).__name__}."
            )

        if "regex_pattern" not in context:
            logger.error(
                "RegexMatcher node context is missing the required 'regex_pattern' key."
            )
            raise ValueError(
                "Missing 'regex_pattern' in context for RegexMatcher node."
            )

        regex_pattern = context["regex_pattern"]
        if not isinstance(regex_pattern, str):
            logger.error(
                "RegexMatcher node received non-string regex_pattern. Expected 'str', got '%s'.",
                type(regex_pattern).__name__,
            )
            raise ValueError(
                f"Invalid 'regex_pattern' type. Expected str, got {type(regex_pattern).__name__}."
            )

        regex_flags = context.get("flags", 0)
        if not isinstance(regex_flags, int):
            logger.warning(
                "RegexMatcher node received non-integer 'flags'. Defaulting to 0."
            )
            regex_flags = 0

        try:
            compiled_pattern = re.compile(regex_pattern, flags=regex_flags)
            matches = compiled_pattern.findall(data)
            logger.debug(
                "RegexMatcher node successfully found %d matches for pattern '%s'.",
                len(matches),
                regex_pattern,
            )
            return matches
        except re.error as e:
            logger.error(
                "RegexMatcher node encountered an invalid regex pattern '%s': %s",
                regex_pattern,
                e,
            )
            raise ValueError(
                f"Invalid regex pattern provided: '{regex_pattern}' - {e}"
            ) from e
        except Exception as e:
            logger.exception(
                "An unexpected error occurred in RegexMatcher node while processing data with pattern '%s'.",
                regex_pattern,
            )
            raise RuntimeError(
                f"An unexpected error occurred during regex matching: {e}"
            ) from e