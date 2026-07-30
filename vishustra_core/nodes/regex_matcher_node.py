import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node expects the input `data` to be a string and looks for
    a 'pattern' key in the `context` dictionary. Optionally, a 'flags'
    key can be provided in the context to modify regex behavior.

    Supported string flags (case-insensitive): "IGNORECASE", "MULTILINE",
    "DOTALL", "UNICODE", "VERBOSE", "ASCII".

    Returns a list of all non-overlapping matches found in the data.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def _parse_flags(self, flag_strings: Union[str, List[str]]) -> int:
        """
        Parses string representation of regex flags into a bitmask.
        """
        if isinstance(flag_strings, str):
            flag_strings = [flag_strings]

        parsed_flags = 0
        for flag_str in flag_strings:
            upper_flag = flag_str.upper()
            if upper_flag == "IGNORECASE":
                parsed_flags |= re.IGNORECASE
            elif upper_flag == "MULTILINE":
                parsed_flags |= re.MULTILINE
            elif upper_flag == "DOTALL":
                parsed_flags |= re.DOTALL
            elif upper_flag == "UNICODE":
                parsed_flags |= re.UNICODE
            elif upper_flag == "VERBOSE":
                parsed_flags |= re.VERBOSE
            elif upper_flag == "ASCII":
                parsed_flags |= re.ASCII
            else:
                logger.warning(
                    f"[{self.node_name}] Unrecognized regex flag '{flag_str}'. Ignoring."
                )
        return parsed_flags

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern.

        Args:
            data: The input data, expected to be a string to be matched against.
            context: A dictionary containing operational parameters:
                - 'pattern' (str): The regular expression pattern to use. (Required)
                - 'flags' (Union[str, List[str], int], optional): Regex flags. Can be
                  a single string (e.g., "IGNORECASE"), a list of strings
                  (e.g., ["IGNORECASE", "MULTILINE"]), or a direct integer bitmask
                  from the `re` module (e.g., re.IGNORECASE). Defaults to 0.

        Returns:
            List[str]: A list of all non-overlapping matches found in the data.
                       Returns an empty list if no matches are found, or if
                       the input data is not a string.

        Raises:
            ValueError: If 'pattern' is missing from context or is not a string.
            re.error: If the provided regex pattern is invalid.
            RuntimeError: For other unexpected internal errors during processing.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is not a string. Cannot perform regex matching. "
                f"Received type: {type(data).__name__}. Returning empty list."
            )
            return []

        pattern_str = context.get("pattern")
        if not isinstance(pattern_str, str):
            error_msg = (
                f"[{self.node_name}] Required 'pattern' key missing or not a string "
                f"in context. Received: {pattern_str!r}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        flags_param = context.get("flags", 0)
        re_flags = 0
        if isinstance(flags_param, int):
            re_flags = flags_param
        elif isinstance(flags_param, (str, list)):
            re_flags = self._parse_flags(flags_param)
        else:
            logger.warning(
                f"[{self.node_name}] Unrecognized type for 'flags' in context: "
                f"{type(flags_param).__name__}. Expected int, str, or list of str. "
                "Defaulting to no flags."
            )

        try:
            matches = re.findall(pattern_str, data, flags=re_flags)
            logger.debug(
                f"[{self.node_name}] Successfully found {len(matches)} matches for pattern '{pattern_str}'."
            )
            return matches
        except re.error as e:
            error_msg = (
                f"[{self.node_name}] Invalid regex pattern '{pattern_str}': {e}"
            )
            logger.error(error_msg)
            raise re.error(error_msg) from e
        except Exception as e:
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during regex matching: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e