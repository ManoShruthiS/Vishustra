import re
import logging
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node that applies a regular expression pattern to the input data
    and extracts matching substrings or specific capture groups.

    This node is configurable to return either the first match or all non-overlapping
    matches, and can be configured to raise an error if no match is found.
    """

    def __init__(
        self,
        pattern: str,
        group_index: int = 0,
        return_all_matches: bool = False,
        raise_on_no_match: bool = False
    ) -> None:
        """
        Initializes the RegexMatcherNode with the specified regex pattern and
        extraction parameters.

        Args:
            pattern (str): The regular expression pattern to compile and use.
                           Must be a non-empty string.
            group_index (int): The index of the capture group to extract.
                               0 for the entire match, 1 for the first captured group, etc.
                               Must be a non-negative integer.
            return_all_matches (bool): If True, the process method will return a list
                                       of all non-overlapping matches/groups found.
                                       If False, it returns only the first match/group.
                                       Defaults to False.
            raise_on_no_match (bool): If True, a ValueError is raised if no match
                                      is found in the input data. If False, the node
                                      returns None (for single match) or an empty list
                                      (for all matches) when no match is found.
                                      Defaults to False.

        Raises:
            ValueError: If the provided pattern is invalid or empty, or if `group_index`
                        is a negative integer.
        """
        if not isinstance(pattern, str) or not pattern:
            logger.error("Invalid pattern provided to RegexMatcherNode: Must be a non-empty string.")
            raise ValueError("Regex pattern must be a non-empty string.")

        try:
            self._compiled_pattern = re.compile(pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}' provided to RegexMatcherNode: {e}")
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

        if not isinstance(group_index, int) or group_index < 0:
            logger.error(f"Invalid group_index provided to RegexMatcherNode: {group_index}. Must be a non-negative integer.")
            raise ValueError("group_index must be a non-negative integer.")

        self._group_index = group_index
        self._return_all_matches = return_all_matches
        self._raise_on_no_match = raise_on_no_match
        logger.debug(
            f"RegexMatcherNode initialized with pattern='{pattern}', "
            f"group_index={group_index}, return_all_matches={return_all_matches}, "
            f"raise_on_no_match={raise_on_no_match}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[str, List[str], None]:
        """
        Applies the configured regex pattern to the input data and extracts matches.

        Args:
            data (Any): The input data to process. This node expects a string input.
            context (Dict[str, Any]): A dictionary containing context-specific information.
                                     This node does not directly use the context dictionary.

        Returns:
            Union[str, List[str], None]:
                - If `return_all_matches` is True, returns a `List[str]` containing
                  all extracted matches/groups. Returns an empty list if no matches
                  are found and `raise_on_no_match` is False.
                - If `return_all_matches` is False, returns a `str` representing the
                  first extracted match/group. Returns `None` if no match is found
                  and `raise_on_no_match` is False.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `raise_on_no_match` is True and no match is found.
            IndexError: If the configured `group_index` is out of bounds for
                        a valid match (i.e., a requested captured group does not exist).
        """
        if not isinstance(data, str):
            logger.error(f"RegexMatcherNode received invalid input data type: Expected str, got {type(data).__name__}.")
            raise TypeError(
                f"RegexMatcherNode requires input 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        log_data_preview = f"'{data[:100]}{'...' if len(data) > 100 else ''}'"
        logger.debug(f"RegexMatcherNode processing data: {log_data_preview}")

        matches_found: List[str] = []
        try:
            for match in self._compiled_pattern.finditer(data):
                try:
                    # group(0) is the entire match. group(1) onwards are captured groups.
                    # len(match.groups()) gives the number of captured groups.
                    # So, valid indices are 0 to len(match.groups()).
                    if self._group_index > len(match.groups()):
                        logger.error(
                            f"Group index {self._group_index} is out of bounds for "
                            f"pattern '{self._compiled_pattern.pattern}' on match "
                            f"'{match.group(0)}'. Only {len(match.groups())} capturing groups available."
                        )
                        raise IndexError(
                            f"Requested group index {self._group_index} is out of bounds. "
                            f"Pattern '{self._compiled_pattern.pattern}' has only "
                            f"{len(match.groups())} capturing groups for match '{match.group(0)}'."
                        )
                    matches_found.append(match.group(self._group_index))
                except IndexError as e:
                    # Re-raise specific IndexErrors regarding group extraction immediately
                    raise e
                except Exception as e:
                    logger.error(f"An unexpected error occurred while extracting group {self._group_index} from a regex match: {e}")
                    raise

        except Exception as e:
            logger.error(f"An unexpected error occurred during regex pattern application: {e}")
            raise

        if not matches_found:
            if self._raise_on_no_match:
                logger.warning(
                    f"No match found for pattern '{self._compiled_pattern.pattern}' in the provided data "
                    f"(raise_on_no_match=True). Data preview: {log_data_preview}"
                )
                raise ValueError(
                    f"No regex match found for pattern '{self._compiled_pattern.pattern}' "
                    f"in the provided data."
                )
            else:
                logger.debug(
                    f"No match found for pattern '{self._compiled_pattern.pattern}' in data. "
                    f"Returning default (empty list or None)."
                )
                return [] if self._return_all_matches else None

        if self._return_all_matches:
            logger.debug(f"Returning all {len(matches_found)} matches found.")
            return matches_found
        else:
            logger.debug(f"Returning the first match: '{matches_found[0]}'")
            return matches_found[0]