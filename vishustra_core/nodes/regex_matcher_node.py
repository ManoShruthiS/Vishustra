import logging
import re
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    This node can be configured to:
    - Search for the first occurrence of a pattern and extract a specific group.
    - Find all non-overlapping occurrences of a pattern.

    Context Parameters:
    - pattern (str, required): The regex pattern to use.
    - flags (int, optional): Bitmask of re flags (e.g., re.IGNORECASE, re.MULTILINE). Default: 0.
    - return_all_matches (bool, optional): If True, returns a list of all matches (re.findall).
                                            If False (default), returns the first match's group (re.search).
    - group_index_or_name (Union[int, str], optional):
        - If `return_all_matches` is False: The index or name of the capturing group to return from the first match.
                                          Defaults to 0 (the entire match).
        - This parameter is ignored when `return_all_matches` is True, as re.findall
          returns either a list of strings or a list of tuples depending on groups.
    - default_value (Any, optional): The value to return if no match is found. Defaults to None.

    Returns:
    - If `return_all_matches` is True: List[str] or List[Tuple[str, ...]] containing all matches.
    - If `return_all_matches` is False: str (the matched group) or the `default_value`.
    - None if input data or pattern is invalid or an error occurs during processing.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data using regex matching based on context parameters.

        Args:
            data (Any): The input data to search within. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing parameters for regex matching.

        Returns:
            Any: The result of the regex operation (matched string, list of strings/tuples, or default value).
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str, got {type(data).__name__}.")
            return None

        pattern: Optional[str] = context.get("pattern")
        if not isinstance(pattern, str):
            logger.error(f"[{self.node_name}] Missing or invalid 'pattern' in context. Expected str.")
            return None

        flags: int = context.get("flags", 0)
        return_all_matches: bool = context.get("return_all_matches", False)
        group_index_or_name: Union[int, str] = context.get("group_index_or_name", 0)
        default_value: Any = context.get("default_value", None)

        logger.debug(
            f"[{self.node_name}] Processing data with pattern: '{pattern}', "
            f"flags: {flags}, return_all_matches: {return_all_matches}, "
            f"group: {group_index_or_name}"
        )

        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            logger.error(f"[{self.node_name}] Invalid regex pattern '{pattern}': {e}")
            return None
        except TypeError as e:
            logger.error(f"[{self.node_name}] Invalid flags provided: {flags}. Error: {e}")
            return None

        if return_all_matches:
            try:
                matches: List[Union[str, Any]] = compiled_pattern.findall(data)
                if not matches:
                    logger.debug(f"[{self.node_name}] No matches found for pattern '{pattern}' using findall.")
                    return default_value
                logger.debug(f"[{self.node_name}] Found {len(matches)} matches using findall.")
                return matches
            except Exception as e:
                logger.error(f"[{self.node_name}] An error occurred during findall operation: {e}")
                return None
        else:
            try:
                match_obj: Optional[re.Match] = compiled_pattern.search(data)
                if not match_obj:
                    logger.debug(f"[{self.node_name}] No match found for pattern '{pattern}' using search.")
                    return default_value
                
                try:
                    result: str = match_obj.group(group_index_or_name)
                    logger.debug(f"[{self.node_name}] Found match: '{match_obj.group(0)}', extracted group '{group_index_or_name}': '{result}'")
                    return result
                except IndexError:
                    logger.error(f"[{self.node_name}] Group index '{group_index_or_name}' is out of range for pattern '{pattern}'.")
                    return default_value
                except KeyError:
                    logger.error(f"[{self.node_name}] Group name '{group_index_or_name}' not found in pattern '{pattern}'.")
                    return default_value
            except Exception as e:
                logger.error(f"[{self.node_name}] An error occurred during search operation: {e}")
                return None
