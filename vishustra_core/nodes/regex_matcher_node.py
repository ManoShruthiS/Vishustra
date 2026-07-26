import re
import logging
from typing import Any, Dict, List, Union, Tuple, Optional

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching
    on input string data. It supports finding all occurrences, searching
    for the first match, or matching from the start of the string based
    on configuration provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying a regular expression pattern.

        The `context` dictionary is used to configure the regex operation:
        - 'pattern' (str): The regular expression pattern string to apply. This is a required parameter.
        - 'match_type' (str, optional): Specifies the type of regex operation to perform.
          Accepts 'findall' (default), 'search', or 'match'.
            - 'findall': Returns a list of all non-overlapping matches. If the pattern
                         has capturing groups, it returns a list of tuples.
            - 'search': Returns the first `re.Match` object found anywhere in the string, or `None` if no match.
            - 'match': Returns an `re.Match` object only if the pattern matches at the beginning
                       of the string, or `None` if no match.
        - 'flags' (int, optional): Bitmask flags for the regex engine, e.g., `re.IGNORECASE | re.DOTALL`.
          Defaults to 0 (no flags).

        Args:
            data (Any): The input data. This node expects `data` to be a string.
            context (Dict[str, Any]): A dictionary containing configuration parameters for the regex matching.

        Returns:
            Any:
                - If 'match_type' is 'findall': `List[str]` or `List[Tuple[str, ...]]` if matches are found,
                  otherwise an empty list.
                - If 'match_type' is 'search' or 'match': An `re.Match` object if a match is found,
                  otherwise `None`.
                - `None`: If an error occurs (e.g., `data` is not a string, 'pattern' is missing or invalid,
                  or an unexpected exception during regex execution).
        """
        if not isinstance(data, str):
            logger.error(
                "RegexMatcherNode received non-string data. Expected string for regex operations. "
                "Node: '%s', Type received: %s", self.node_name, type(data)
            )
            return None

        pattern_str = context.get('pattern')
        if not isinstance(pattern_str, str) or not pattern_str:
            logger.error(
                "RegexMatcherNode requires a valid 'pattern' (non-empty string) in the context. "
                "Node: '%s', Received pattern: %s", self.node_name, pattern_str
            )
            return None

        match_type = context.get('match_type', 'findall').lower()
        flags = context.get('flags', 0)

        if not isinstance(flags, int):
            logger.warning(
                "RegexMatcherNode received non-integer 'flags' in context for node '%s'. "
                "Defaulting to 0. Received: %s", self.node_name, flags
            )
            flags = 0

        try:
            compiled_pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.error(
                "RegexMatcherNode encountered an invalid regex pattern '%s' for node '%s': %s",
                pattern_str, self.node_name, e
            )
            return None

        try:
            if match_type == 'findall':
                return compiled_pattern.findall(data)
            elif match_type == 'search':
                return compiled_pattern.search(data)
            elif match_type == 'match':
                return compiled_pattern.match(data)
            else:
                logger.warning(
                    "RegexMatcherNode received unknown 'match_type': '%s' for node '%s'. "
                    "Defaulting to 'findall'.", match_type, self.node_name
                )
                return compiled_pattern.findall(data)
        except Exception as e:
            # Catch any other potential runtime issues during regex application
            logger.error(
                "RegexMatcherNode failed during '%s' operation with pattern '%s' on data "
                "(first 100 chars): '%s' for node '%s': %s",
                match_type, pattern_str, data[:100], self.node_name, e
            )
            return None
