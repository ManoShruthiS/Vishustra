import re
import logging
from typing import Any, Dict, List, Optional

# Assuming BaseNode is available at this path in the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node can be configured to:
    - Search for the first occurrence of a pattern and return a specific capturing group.
    - Find all non-overlapping occurrences of a pattern and return a list of 
      specific capturing groups for each match.

    Initialization Parameters:
        pattern (str): The regular expression pattern string to be compiled.
        group_index (int): The index of the capturing group to return. 
                           Group 0 means the entire match. Defaults to 0.
        return_all_matches (bool): If True, all non-overlapping matches will be returned 
                                   as a list of strings. If False, only the first match 
                                   is returned as a string or None. Defaults to False.
        flags (int): Regular expression flags (e.g., re.IGNORECASE, re.MULTILINE).
                     Defaults to 0 (no flags).

    Raises:
        ValueError: If the provided pattern is not a non-empty string.
        re.error: If the provided pattern is an invalid regular expression.
    """

    def __init__(self, pattern: str, group_index: int = 0, return_all_matches: bool = False, flags: int = 0):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Regex pattern must be a non-empty string.")
        
        self._pattern = pattern
        self._group_index = group_index
        self._return_all_matches = return_all_matches
        self._flags = flags
        
        try:
            self._compiled_pattern = re.compile(self._pattern, self._flags)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{self._pattern}' provided to RegexMatcherNode: {e}")
            raise

        logger.debug(f"RegexMatcherNode initialized with pattern: '{pattern}', "
                     f"group_index: {group_index}, return_all_matches: {return_all_matches}, "
                     f"flags: {flags}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Optional[Any]:
        """
        Processes the input data by applying the configured regex pattern.

        Args:
            data (Any): The input data to match against. This node expects a string.
            context (Dict[str, Any]): A dictionary containing additional context 
                                       information for processing (not directly used by this node).

        Returns:
            Optional[Any]: 
                - If `return_all_matches` is False: The matched string from the specified 
                  group, or None if no match is found, data is not a string, or an error occurs.
                - If `return_all_matches` is True: A list of matched strings from the 
                  specified group for all matches found. Returns an empty list if no matches,
                  data is not a string, or an error occurs.
        """
        if not isinstance(data, str):
            logger.warning(f"Input data for RegexMatcherNode must be a string, "
                           f"but received type: {type(data).__name__}. Returning {'[]' if self._return_all_matches else 'None'}.")
            return [] if self._return_all_matches else None
        
        try:
            if self._return_all_matches:
                matches: List[str] = []
                for match in self._compiled_pattern.finditer(data):
                    try:
                        matches.append(match.group(self._group_index))
                    except IndexError:
                        logger.warning(f"Group index {self._group_index} out of range for a match "
                                       f"found by pattern '{self._pattern}'. Skipping this specific match.")
                        # Continue to find other matches
                logger.debug(f"Found {len(matches)} total matches for pattern '{self._pattern}'.")
                return matches
            else:
                match = self._compiled_pattern.search(data)
                if match:
                    try:
                        result = match.group(self._group_index)
                        logger.debug(f"First match found for pattern '{self._pattern}': '{result}'.")
                        return result
                    except IndexError:
                        # Log partial data to prevent excessively long log messages
                        log_data_snippet = data[:200] + ('...' if len(data) > 200 else '')
                        logger.error(f"Group index {self._group_index} out of range for "
                                      f"pattern '{self._pattern}' in data '{log_data_snippet}'. Returning None.")
                        return None
                else:
                    logger.debug(f"No match found for pattern '{self._pattern}' in data.")
                    return None
        except Exception as e:
            # Catch any other unexpected errors during the regex processing
            logger.error(f"An unexpected error occurred during regex matching with "
                         f"pattern '{self._pattern}': {e}", exc_info=True)
            return [] if self._return_all_matches else None
