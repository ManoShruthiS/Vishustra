import re
import logging
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node that applies a regular expression pattern to the input data.
    It can return the first matched string (or a specific group from it) or all
    non-overlapping matches found, always respecting the specified group index.
    
    Configuration for the node is expected in the 'context' dictionary:
    - 'regex_pattern' (str): The regular expression pattern to use. (REQUIRED)
    - 'return_all_matches' (bool, optional): If True, returns a list of strings,
      where each string is the content of the specified 'group_index' from each
      non-overlapping match. If False, returns the content of 'group_index'
      from the first match found. Defaults to False.
    - 'group_index' (int, optional): Specifies which capturing group (0-indexed)
      to return. Group 0 means the entire match. Defaults to 0.
    - 'flags' (int, optional): Bitmask of re flags (e.g., re.IGNORECASE, re.MULTILINE).
      Defaults to 0 (no flags).
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[str, List[str], None]:
        """
        Applies a regular expression to the input data based on context configuration.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing configuration parameters
                                      for the regex operation, including:
                                      - 'regex_pattern' (str): The regex pattern.
                                      - 'return_all_matches' (bool, optional): Defaults to False.
                                      - 'group_index' (int, optional): Defaults to 0.
                                      - 'flags' (int, optional): Defaults to 0.

        Returns:
            Union[str, List[str], None]:
                - If 'return_all_matches' is True: A list of strings, each being
                  the content of the specified 'group_index' from each match.
                  Returns an empty list if no matches are found.
                - If 'return_all_matches' is False: The content of 'group_index'
                  from the first match. Returns None if no match is found.

        Raises:
            ValueError: If 'regex_pattern' is missing from context or is not a string.
            TypeError: If input 'data' is not a string.
            re.error: If the 'regex_pattern' provided is invalid.
            IndexError: If 'group_index' is out of bounds for a specific match's groups.
        """
        logger.debug(f"[{self.node_name}] Processing data with context: {context}")

        # --- Input Data Validation ---
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Input data must be a string, but received type: {type(data)}. Raising TypeError.")
            raise TypeError(f"Input data for RegexMatcherNode must be a string, received {type(data)}")

        # --- Context Parameter Retrieval and Validation ---
        regex_pattern: Optional[str] = context.get('regex_pattern')
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error(f"[{self.node_name}] 'regex_pattern' is missing or invalid in context. Received: {regex_pattern}. Raising ValueError.")
            raise ValueError("Context must contain a valid 'regex_pattern' (non-empty string).")

        return_all_matches: bool = context.get('return_all_matches', False)
        if not isinstance(return_all_matches, bool):
            logger.warning(f"[{self.node_name}] 'return_all_matches' in context should be a boolean. Received {type(return_all_matches)}. Defaulting to False.")
            return_all_matches = False
        
        group_index: int = context.get('group_index', 0)
        if not isinstance(group_index, int) or group_index < 0:
            logger.warning(f"[{self.node_name}] 'group_index' in context should be a non-negative integer. Received {group_index}. Defaulting to 0.")
            group_index = 0

        flags: int = context.get('flags', 0)
        if not isinstance(flags, int):
             logger.warning(f"[{self.node_name}] 'flags' in context should be an integer bitmask. Received {type(flags)}. Defaulting to 0.")
             flags = 0

        # --- Regex Compilation ---
        try:
            compiled_pattern = re.compile(regex_pattern, flags)
            logger.debug(f"[{self.node_name}] Successfully compiled regex pattern: '{regex_pattern}' with flags: {flags}")
        except re.error as e:
            logger.exception(f"[{self.node_name}] Invalid regex pattern provided: '{regex_pattern}'. Raising re.error.")
            raise re.error(f"Invalid regex pattern: {regex_pattern}") from e

        # --- Regex Matching Logic ---
        if return_all_matches:
            extracted_matches: List[str] = []
            for match in compiled_pattern.finditer(data):
                try:
                    # group_index 0 refers to the entire match, which is always valid if a match object exists.
                    # For group_index > 0, it must be less than or equal to the total number of *capturing* groups.
                    if group_index > 0 and group_index > compiled_pattern.groups:
                        logger.warning(f"[{self.node_name}] Group index {group_index} is out of bounds for pattern '{regex_pattern}' (only {compiled_pattern.groups} capturing groups available). Skipping this match for consistency.")
                        continue
                    
                    extracted_matches.append(match.group(group_index))
                except IndexError as e:
                    # This catch is a safeguard for unexpected edge cases, as the above check should prevent most.
                    logger.warning(f"[{self.node_name}] IndexError accessing group {group_index} for a match. Pattern: '{regex_pattern}'. Error: {e}. Skipping this match.")
                    continue

            logger.info(f"[{self.node_name}] Found {len(extracted_matches)} matches using group index {group_index} (return_all_matches=True).")
            return extracted_matches
        else:
            match = compiled_pattern.search(data)
            if match:
                try:
                    # group_index 0 refers to the entire match, always valid if a match object exists.
                    # For group_index > 0, it must be less than or equal to the total number of *capturing* groups.
                    if group_index > 0 and group_index > compiled_pattern.groups:
                        logger.error(f"[{self.node_name}] Group index {group_index} is out of bounds for pattern '{regex_pattern}' (only {compiled_pattern.groups} capturing groups available). Raising IndexError.")
                        raise IndexError(f"Group index {group_index} out of bounds for pattern '{regex_pattern}'. Only {compiled_pattern.groups} capturing groups available.")
                    
                    result = match.group(group_index)
                    logger.info(f"[{self.node_name}] Found first match for pattern, returning group {group_index}.")
                    return result
                except IndexError as e:
                    logger.exception(f"[{self.node_name}] IndexError when trying to access group {group_index} for the first match. Pattern: '{regex_pattern}'. Error: {e}. Raising IndexError.")
                    raise IndexError(f"Group index {group_index} out of bounds for match.") from e
            else:
                logger.info(f"[{self.node_name}] No match found for pattern.")
                return None