import logging
import re
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node for performing regex pattern matching and extraction
    on input data.

    This node offers flexible control over regex operations:
    - Extracting either the first match or all non-overlapping matches.
    - Specifying a particular capture group by index or name.
    - Choosing between `re.search` (find anywhere in string) or `re.match` (find only at string start).
    - Custom regex flags for advanced matching behavior.

    Context parameters:
    - 'regex_pattern' (str): The regular expression pattern to use. (Required)
    - 'return_all_matches' (bool, optional): If `True`, the node uses `re.finditer`
      to return a list of all non-overlapping matches. If `False` (default), it returns
      only the first match found.
    - 'group_index' (int, optional): The index of the capture group to return.
      Defaults to 0 (the entire match). This parameter is ignored if 'group_name' is provided.
    - 'group_name' (str, optional): The name of the capture group to return.
      If provided, it takes precedence over 'group_index'.
    - 'match_type' (str, optional): Specifies the matching method when
      'return_all_matches' is `False`. Can be 'search' (default, uses `re.search`)
      or 'match' (uses `re.match`). This parameter is ignored if 'return_all_matches' is `True`.
    - 'flags' (int, optional): A bitmask of `re` flags (e.g., `re.IGNORECASE | re.MULTILINE`).
      Defaults to 0 (no flags).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[Optional[str], List[str]]:
        """
        Processes the input data by applying a regular expression pattern
        and extracting matched content.

        Args:
            data: The input data to be processed, expected to be a string.
            context: A dictionary containing configuration parameters for the
                     regex operation, as described in the class docstring.

        Returns:
            - If 'return_all_matches' is `True`: A `List[str]` where each string
              is an extracted group from a match. Returns an empty list if no matches
              are found or no groups can be extracted for valid matches.
            - If 'return_all_matches' is `False`: An `Optional[str]` representing
              the extracted group from the first match. Returns `None` if no match
              is found.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'regex_pattern' is missing, invalid, or if group
                        extraction for a single match fails critically.
        """
        if not isinstance(data, str):
            logger.error(
                "RegexMatcherNode received non-string data. Expected type 'str', but got '%s'.",
                type(data).__name__
            )
            raise TypeError(f"RegexMatcherNode expects 'data' to be a string, but received {type(data).__name__}.")

        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error(
                "RegexMatcherNode requires a valid 'regex_pattern' (a non-empty string) in the context. "
                "Received: '%s'", regex_pattern
            )
            raise ValueError("Missing or invalid 'regex_pattern' in context.")

        return_all_matches: bool = context.get("return_all_matches", False)
        group_index: int = context.get("group_index", 0)
        group_name: Optional[str] = context.get("group_name")
        match_type: str = context.get("match_type", "search").lower()
        flags: int = context.get("flags", 0)

        try:
            compiled_pattern = re.compile(regex_pattern, flags)
        except re.error as e:
            logger.error("Failed to compile regex pattern '%s': %s", regex_pattern, e)
            raise ValueError(f"Invalid regex pattern: '{regex_pattern}'. Error: {e}") from e

        if return_all_matches:
            results: List[str] = []
            # When 'return_all_matches' is True, re.finditer is used to find all occurrences.
            # 'match_type' (search/match) is not directly applicable here as `re.match` is
            # anchored to the beginning of the string, which conflicts with "all matches"
            # across the string. `re.finditer` effectively performs `re.search` for all matches.
            for match in compiled_pattern.finditer(data):
                try:
                    if group_name is not None:
                        results.append(match.group(group_name))
                    else:
                        results.append(match.group(group_index))
                except (IndexError, KeyError) as e:
                    # Log a warning and skip this specific match if its group extraction fails.
                    # This allows other valid matches to still be returned.
                    matched_text_snippet = (match.group(0) or "<empty match>")[:50]
                    logger.warning(
                        "RegexMatcherNode: Failed to extract group from a match (pattern: '%s', data snippet: '%s...'). "
                        "Group index/name requested: %s/%s. Error: %s. Skipping this match.",
                        regex_pattern, matched_text_snippet, group_index, group_name, e
                    )
            return results
        else:
            # Return only the first match found
            match = None
            if match_type == "search":
                match = compiled_pattern.search(data)
            elif match_type == "match":
                match = compiled_pattern.match(data)
            else:
                logger.warning(
                    "Unknown 'match_type' in context: '%s'. Defaulting to 'search' behavior for the first match.",
                    match_type
                )
                match = compiled_pattern.search(data)  # Fallback to re.search

            if match:
                try:
                    if group_name is not None:
                        return match.group(group_name)
                    else:
                        return match.group(group_index)
                except (IndexError, KeyError) as e:
                    # If group extraction fails for the single requested match, it's a critical error.
                    matched_text_snippet = (match.group(0) or "<empty match>")[:50]
                    logger.error(
                        "RegexMatcherNode: Failed to extract specified group from the first match "
                        "(pattern: '%s', data snippet: '%s...'). Group index/name requested: %s/%s. Error: %s",
                        regex_pattern, matched_text_snippet, group_index, group_name, e
                    )
                    raise ValueError(f"Failed to extract specified group from match: {e}") from e
            return None