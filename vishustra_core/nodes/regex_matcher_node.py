import logging
import re
from typing import Any, Dict, List, Optional, Union

# Assuming BaseNode is defined in this module path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node extracts substrings from the input data (expected to be a string)
    based on a provided regular expression pattern from the context.
    It can be configured to return either all non-overlapping matches or just
    the first match found.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], Optional[str]]:
        """
        Processes the input data by applying a regular expression pattern.

        The `context` dictionary must contain:
        - 'regex_pattern' (str): The regular expression pattern to use.

        Optional keys in `context`:
        - 'flags' (int, default: 0): Regex flags, e.g., re.IGNORECASE, re.MULTILINE.
          These can be combined using the bitwise OR operator (e.g., `re.IGNORECASE | re.DOTALL`).
        - 'return_first_match_only' (bool, default: False): If True, returns
          only the content of the specified group from the first match (or None if no match).
          If False, returns a list of the content of the specified group from all
          non-overlapping matches.
        - 'group_index' (int, default: 0): Specifies which group to return from a match.
          0 corresponds to the entire matched string, 1 to the first capturing group, and so on.

        Args:
            data: The input data, expected to be a string for regex matching.
            context: A dictionary containing processing parameters.

        Returns:
            Union[List[str], Optional[str]]:
            - A list of strings if 'return_first_match_only' is False. Each element
              is the content of the specified 'group_index' from each match.
              Returns an empty list if no matches are found.
            - A string (the content of the specified group from the first match)
              if 'return_first_match_only' is True. Returns None if no match is found.

        Raises:
            TypeError: If `data` is not a string.
            KeyError: If 'regex_pattern' is missing from the context.
            re.error: If the provided 'regex_pattern' is an invalid regular expression.
        """
        if not isinstance(data, str):
            logger.error(f"Input data for {self.node_name} must be a string, but received {type(data).__name__}.")
            raise TypeError(f"Input data must be a string for {self.node_name} processing.")

        regex_pattern = context.get('regex_pattern')
        if not regex_pattern:
            logger.error(f"Missing 'regex_pattern' in context for {self.node_name}.")
            raise KeyError(f"'regex_pattern' is required in context for {self.node_name}.")

        flags = context.get('flags', 0)
        return_first_match_only = context.get('return_first_match_only', False)
        group_index = context.get('group_index', 0)

        logger.debug(
            f"[{self.node_name}] Processing data with pattern: '{regex_pattern}', "
            f"flags: {flags}, return_first_only: {return_first_match_only}, "
            f"group_index: {group_index}"
        )

        try:
            if return_first_match_only:
                match = re.search(regex_pattern, data, flags)
                if match:
                    try:
                        result = match.group(group_index)
                        logger.info(f"[{self.node_name}] Found first match (group {group_index}): '{result}'")
                        return result
                    except IndexError:
                        logger.warning(
                            f"[{self.node_name}] Group index {group_index} out of range "
                            f"for pattern '{regex_pattern}' on data excerpt: '{data[:50]}...'. "
                            "Returning full match (group 0) instead for the first match."
                        )
                        return match.group(0) # Fallback to full match if group_index is invalid
                else:
                    logger.debug(f"[{self.node_name}] No match found for pattern '{regex_pattern}'.")
                    return None
            else: # Return all matches
                matches: List[str] = []
                for match in re.finditer(regex_pattern, data, flags):
                    try:
                        matches.append(match.group(group_index))
                    except IndexError:
                        logger.warning(
                            f"[{self.node_name}] Group index {group_index} out of range "
                            f"for pattern '{regex_pattern}' on match: '{match.group(0)}'. "
                            "Defaulting to full match (group 0) for this instance."
                        )
                        # Fallback to full match (group 0) if specified group_index is invalid
                        matches.append(match.group(0))
                logger.info(f"[{self.node_name}] Found {len(matches)} matches.")
                logger.debug(f"[{self.node_name}] All matches (group {group_index}): {matches}")
                return matches
        except re.error as e:
            logger.error(f"[{self.node_name}] Invalid regex pattern '{regex_pattern}': {e}")
            raise re.error(f"Invalid regex pattern provided for {self.node_name}: {e}")
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected error occurred during processing: {e}",
                exc_info=True
            )
            raise # Re-raise unexpected exceptions to propagate upstream