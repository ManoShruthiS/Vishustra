import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple

# Assuming BaseNode is available at this path relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching on input data.

    This node extracts parts of the input `data` (expected to be a string) based on
    a provided regular expression pattern and matching mode.

    Configuration in context:
    - 'regex_pattern' (str): The regular expression pattern to match. (Required)
    - 'regex_mode' (str): The matching mode.
        - 'findall' (default): Returns a list of all non-overlapping matches.
          If the pattern contains capturing groups, it returns a list of tuples
          where each tuple contains the captured groups.
        - 'search': Returns the first match found. If 'regex_group' is specified,
          it returns the content of that specific group. Returns None if no match.
    - 'regex_group' (int): (Only applicable for 'search' mode)
      The index of the capturing group to return. 0 means the entire match. Defaults to 0.
      If the specified group index is out of bounds, None is returned.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[Union[str, Tuple[str, ...]]], str, None]:
        """
        Processes the input data by applying a regex pattern defined in the context.

        Args:
            data (Any): The input data, expected to be a string for regex operations.
            context (Dict[str, Any]): A dictionary containing node-specific configuration
                                      and shared context. Must contain 'regex_pattern'.

        Returns:
            Union[List[Union[str, Tuple[str, ...]]], str, None]:
                - If 'regex_mode' is 'findall': A list of matching strings or tuples.
                  Returns an empty list if no matches are found or if input is invalid.
                - If 'regex_mode' is 'search': The matched string (or a specific group) or None.
                  Returns None if no match is found or if input/configuration is invalid.
        """
        # Determine the expected empty result based on mode for consistent error returns
        default_empty_result = [] if context.get('regex_mode', 'findall').lower() == 'findall' else None

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is not a string. "
                f"Expected string for regex matching, got {type(data).__name__}. Returning empty result."
            )
            return default_empty_result

        regex_pattern = context.get('regex_pattern')
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error(
                f"[{self.node_name}] Missing or invalid 'regex_pattern' in context. "
                "A non-empty string pattern is required. Returning empty result."
            )
            return default_empty_result

        regex_mode = context.get('regex_mode', 'findall').lower()
        regex_group = context.get('regex_group', 0)

        if not isinstance(regex_group, int):
            logger.warning(
                f"[{self.node_name}] 'regex_group' in context is not an integer. "
                f"Falling back to default group 0 for search mode. Got {type(regex_group).__name__}."
            )
            regex_group = 0

        try:
            compiled_pattern = re.compile(regex_pattern)
        except re.error as e:
            logger.error(
                f"[{self.node_name}] Invalid regex pattern '{regex_pattern}': {e}. Returning empty result."
            )
            return default_empty_result

        if regex_mode == 'findall':
            matches = compiled_pattern.findall(data)
            logger.debug(
                f"[{self.node_name}] Performed findall with pattern '{regex_pattern}'. "
                f"Found {len(matches)} matches."
            )
            return matches
        elif regex_mode == 'search':
            match = compiled_pattern.search(data)
            if match:
                try:
                    result = match.group(regex_group)
                    logger.debug(
                        f"[{self.node_name}] Performed search with pattern '{regex_pattern}', group {regex_group}. "
                        f"Found match: '{result}'."
                    )
                    return result
                except IndexError:
                    logger.error(
                        f"[{self.node_name}] Specified regex group {regex_group} is out of bounds "
                        f"for pattern '{regex_pattern}' and matched text. Returning None."
                    )
                    return None
            else:
                logger.debug(
                    f"[{self.node_name}] Performed search with pattern '{regex_pattern}'. "
                    "No match found. Returning None."
                )
                return None
        else:
            logger.error(
                f"[{self.node_name}] Invalid 'regex_mode' '{regex_mode}' in context. "
                "Supported modes are 'findall' and 'search'. Returning empty result."
            )
            return default_empty_result