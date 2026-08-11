import logging
import re
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching
    on input data.

    This node provides functionality to extract patterns, find the first match,
    or verify a match at the beginning of the string, configurable through
    the context dictionary. It supports standard regex flags.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], Optional[str]]:
        """
        Processes the input data by applying a regular expression pattern.

        The behavior of the regex operation is determined by the 'context' dictionary.

        Args:
            data: The string data to perform regex matching on.
                  If `data` is not a string, an error is logged, and an empty list or None is returned.
            context: A dictionary containing configuration for the regex operation:
                     - 'pattern' (str): The regular expression pattern to use. (Required)
                       If missing or not a string, an error is logged.
                     - 'flags' (Union[int, str], optional): Regex flags. Can be an integer
                       (e.g., `re.IGNORECASE | re.MULTILINE`) or a comma-separated string
                       of flag names (e.g., "IGNORECASE,DOTALL"). Case-insensitive.
                       Defaults to 0 (no flags). Unknown string flags are warned and ignored.
                     - 'match_method' (str, optional): The method to use:
                       "findall" (default): Returns all non-overlapping matches as a list of strings.
                       "search": Scans through the string looking for the first location where
                                 the regex produces a match. Returns the full matched string, or None.
                       "match": Attempts to match the regex only at the beginning of the string.
                                Returns the full matched string, or None.
                       If an invalid method is specified, "findall" is used as a fallback.

        Returns:
            - If 'match_method' is "findall": A `List[str]` containing all found matches.
            - If 'match_method' is "search" or "match": An `Optional[str]` representing the
              full matched string if found, otherwise `None`.
            - Returns an empty list or `None` and logs an error if input data is invalid,
              or if the pattern is missing/invalid, preventing node execution.
        """
        # 1. Validate input data type
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'. Returning empty result."
            )
            # Return type depends on the expected method for robust type hinting consistency
            return [] if context.get('match_method', 'findall').lower() == 'findall' else None

        # 2. Extract and validate regex pattern from context
        pattern_str = context.get("pattern")
        if not isinstance(pattern_str, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'pattern' in context. Expected 'str'. "
                "Returning empty result."
            )
            return [] if context.get('match_method', 'findall').lower() == 'findall' else None

        # 3. Process regex flags
        flags = 0
        context_flags = context.get("flags", 0)
        if isinstance(context_flags, int):
            flags = context_flags
        elif isinstance(context_flags, str):
            flag_map = {
                "IGNORECASE": re.IGNORECASE, "MULTILINE": re.MULTILINE,
                "DOTALL": re.DOTALL, "VERBOSE": re.VERBOSE,
                "ASCII": re.ASCII, "UNICODE": re.UNICODE, "LOCALE": re.LOCALE,
            }
            # Support multiple flags via comma-separated string
            for flag_name in context_flags.upper().split(','):
                flag_name = flag_name.strip()
                if flag_name in flag_map:
                    flags |= flag_map[flag_name]
                else:
                    logger.warning(
                        f"[{self.node_name}] Unknown regex flag '{flag_name}' specified. Ignoring."
                    )
        else:
            logger.warning(
                f"[{self.node_name}] Invalid 'flags' type in context. Expected 'int' or 'str', "
                f"got '{type(context_flags).__name__}'. Defaulting to no flags (0)."
            )

        # 4. Determine matching method
        match_method = context.get("match_method", "findall").lower()

        # 5. Compile the regex pattern
        try:
            compiled_pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.error(
                f"[{self.node_name}] Invalid regex pattern '{pattern_str}' with flags {flags}: {e}. "
                "Returning empty result."
            )
            return [] if match_method == 'findall' else None

        # 6. Execute the specified matching method
        if match_method == "findall":
            return compiled_pattern.findall(data)
        elif match_method == "search":
            match_obj = compiled_pattern.search(data)
            return match_obj.group(0) if match_obj else None
        elif match_method == "match":
            match_obj = compiled_pattern.match(data)
            return match_obj.group(0) if match_obj else None
        else:
            logger.error(
                f"[{self.node_name}] Invalid 'match_method' specified: '{match_method}'. "
                "Supported methods are 'findall', 'search', 'match'. Falling back to 'findall'."
            )
            return compiled_pattern.findall(data) # Fallback to default
