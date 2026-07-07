import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching
    on input data.

    This node expects the input `data` to be a string and requires a
    `regex_pattern` string in the `context` dictionary to define the
    pattern to match. It returns all non-overlapping matches found in
    the data.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern
        from the context.

        Args:
            data: The input string on which the regex matching will be performed.
                  Must be of type `str`.
            context: A dictionary containing operational parameters for the node.
                     Must include a key 'regex_pattern' with a string value
                     representing the regular expression.

        Returns:
            A list of strings, where each string is a non-overlapping match
            found by the regex pattern in the input data. Returns an empty
            list if no matches are found.

        Raises:
            TypeError: If the `data` input is not a string.
            ValueError: If 'regex_pattern' is missing from the `context`,
                        is not a string, or is an invalid regular expression.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Input data type error. Expected string, but received %s.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for regex matching, "
                f"but received {type(data).__name__}."
            )

        regex_pattern = context.get("regex_pattern")

        if regex_pattern is None:
            logger.error(
                "[%s] Missing configuration: 'regex_pattern' is not provided in the context.",
                self.node_name,
            )
            raise ValueError(
                f"[{self.node_name}] 'regex_pattern' must be provided in the context."
            )

        if not isinstance(regex_pattern, str):
            logger.error(
                "[%s] Configuration error: 'regex_pattern' in context is not a string. Received %s.",
                self.node_name,
                type(regex_pattern).__name__,
            )
            raise ValueError(
                f"[{self.node_name}] 'regex_pattern' in context must be a string, "
                f"but received {type(regex_pattern).__name__}."
            )

        try:
            compiled_pattern = re.compile(regex_pattern)
        except re.error as e:
            logger.error(
                "[%s] Invalid regex pattern provided: '%s'. Error: %s",
                self.node_name,
                regex_pattern,
                e,
            )
            raise ValueError(
                f"[{self.node_name}] Invalid regex pattern provided: {e}"
            ) from e

        matches = compiled_pattern.findall(data)
        logger.debug(
            "[%s] Successfully found %d matches for pattern '%s' in data (first 100 chars: '%s...').",
            self.node_name,
            len(matches),
            regex_pattern,
            data[:100],
        )

        return matches