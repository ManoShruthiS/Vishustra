from vishustra_core.nodes.base_node import BaseNode
import re
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node that performs regular expression matching on input data.

    This node is designed to extract specific patterns from text-based input.
    It supports returning either the first match or all matches found,
    and allows for extraction of specific capturing groups (by index or name)
    from the regex pattern.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
        """
        Processes the input data by applying a regular expression to extract information.

        The `context` dictionary must contain configuration parameters for the regex operation.

        Expected `context` keys:
        - `regex_pattern` (str): The regular expression pattern to use. This is a required parameter.
        - `regex_flags` (int, optional): Flags for the regex engine (e.g., `re.IGNORECASE`, `re.MULTILINE`).
                                         Defaults to `0` (no flags).
        - `return_all_matches` (bool, optional): If `True`, the node attempts to find and return
                                                  a list of all non-overlapping matches.
                                                  If `False`, it returns only the first match found.
                                                  Defaults to `False`.
        - `regex_group` (Union[int, str], optional): The specific capturing group to extract from each match.
                                                      Can be an integer index (e.g., `1` for the first group)
                                                      or a string name for a named group (e.g., `'name'`).
                                                      Defaults to `0`, which represents the entire match.

        Args:
            data: The input data, which is expected to be a string or convertible to a string.
            context: A dictionary containing configuration settings for the regex operation.

        Returns:
            - If `return_all_matches` is `False`: The extracted string for the first match's
              specified group, or `None` if no match is found.
            - If `return_all_matches` is `True`: A list of extracted strings for all
              matches, or an empty list (`[]`) if no matches are found.

        Raises:
            TypeError: If the input `data` cannot be converted to a string.
            ValueError: If `regex_pattern` is missing, invalid, or if `regex_group` is negative.
            IndexError: If an integer `regex_group` is specified but does not exist in the pattern's match.
            KeyError: If a named `regex_group` is specified but does not exist in the pattern's match.
        """
        processed_data: str
        if not isinstance(data, str):
            try:
                processed_data = str(data)
                logger.warning(
                    f"Input data for RegexMatcherNode is not a string (type: {type(data)}). "
                    f"Attempting to convert to string: '{processed_data}'."
                )
            except Exception as e:
                logger.error(
                    f"RegexMatcherNode received non-string data ({type(data)}) and failed to convert it. Error: {e}"
                )
                raise TypeError(
                    f"RegexMatcherNode requires string data or data convertible to string. "
                    f"Received type: {type(data)}."
                ) from e
        else:
            processed_data = data

        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error(f"Missing or invalid 'regex_pattern' in context: {regex_pattern}")
            raise ValueError("Context must contain a non-empty string 'regex_pattern'.")

        regex_flags = context.get("regex_flags", 0)
        if not isinstance(regex_flags, int):
            logger.warning(
                f"Invalid 'regex_flags' type in context ({type(regex_flags)}). "
                f"Defaulting to 0. Value provided: {regex_flags}."
            )
            regex_flags = 0

        return_all_matches = context.get("return_all_matches", False)
        if not isinstance(return_all_matches, bool):
            logger.warning(
                f"Invalid 'return_all_matches' type in context ({type(return_all_matches)}). "
                f"Defaulting to False. Value provided: {return_all_matches}."
            )
            return_all_matches = False

        regex_group = context.get("regex_group", 0)
        if not isinstance(regex_group, (int, str)):
            logger.warning(
                f"Invalid 'regex_group' type in context ({type(regex_group)}). "
                f"Defaulting to 0. Value provided: {regex_group}."
            )
            regex_group = 0
        if isinstance(regex_group, int) and regex_group < 0:
            logger.error(f"Invalid 'regex_group' value: {regex_group}. Group index must be non-negative.")
            raise ValueError("Regex group index must be non-negative.")

        try:
            compiled_pattern = re.compile(regex_pattern, regex_flags)
            logger.debug(f"Compiled regex pattern: '{regex_pattern}' with flags: {regex_flags}.")
        except re.error as e:
            logger.error(f"Invalid regex pattern provided: '{regex_pattern}'. Error: {e}")
            raise ValueError(f"Invalid regex pattern: {e}") from e

        if return_all_matches:
            logger.debug(f"Attempting to find all matches for pattern '{regex_pattern}' in data and extract group '{regex_group}'.")
            extracted_results: List[str] = []
            
            for match in compiled_pattern.finditer(processed_data):
                try:
                    extracted_results.append(match.group(regex_group))
                except (IndexError, KeyError) as e:
                    logger.warning(
                        f"Group '{regex_group}' not found in one of the matches for pattern '{regex_pattern}'. "
                        f"Match details: groups={match.groups()}, dict={match.groupdict()}. Error: {e}. "
                        "Skipping this particular match result."
                    )
            
            if not extracted_results and compiled_pattern.search(processed_data):
                 logger.warning(
                     f"No results extracted for group '{regex_group}' despite matches being found for pattern "
                     f"'{regex_pattern}'. This might indicate an invalid group specification for the pattern or "
                     "conditional groups that did not capture anything."
                 )
            elif not extracted_results:
                logger.debug("No matches found for pattern.")
            else:
                logger.info(f"Found {len(extracted_results)} matches for pattern '{regex_pattern}'.")
            
            return extracted_results

        else: # Return first match only
            logger.debug(f"Attempting to find first match for pattern '{regex_pattern}' in data and extract group '{regex_group}'.")
            match = compiled_pattern.search(processed_data)
            if match:
                try:
                    result = match.group(regex_group)
                    logger.info(
                        f"First match found for pattern '{regex_pattern}', "
                        f"extracted group '{regex_group}': '{result}'."
                    )
                    return result
                except IndexError:
                    logger.error(
                        f"Group index {regex_group} not found in regex pattern '{regex_pattern}'. "
                        f"Available groups: {match.groups()}."
                    )
                    raise IndexError(
                        f"Requested group index {regex_group} does not exist in the pattern's match for '{regex_pattern}'."
                    )
                except KeyError:
                    logger.error(
                        f"Named group '{regex_group}' not found in regex pattern '{regex_pattern}'. "
                        f"Available named groups: {match.groupdict().keys()}."
                    )
                    raise KeyError(
                        f"Requested named group '{regex_group}' does not exist in the pattern's match for '{regex_pattern}'."
                    )
            else:
                logger.debug(f"No match found for pattern '{regex_pattern}'.")
                return None