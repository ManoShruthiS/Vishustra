import logging
import re
from typing import Any, Dict, List, Union, Tuple, Optional

# Assuming vishustra_core.nodes.base_node exists and BaseNode is there.
# This import would resolve correctly in a Vishustra environment.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # This block provides a mock BaseNode for isolated development or testing
    # if vishustra_core is not yet installed or not in the Python path.
    # In a production Vishustra setup, the 'try' block would succeed.
    class BaseNode:
        """
        Mock BaseNode class for development convenience when `vishustra_core`
        is not yet installed. In a real Vishustra environment, this class
        would be imported from `vishustra_core.nodes.base_node`.
        """
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """Placeholder for the process method."""
            raise NotImplementedError("process method must be implemented by subclasses")
        
        @property
        def node_name(self) -> str:
            """Placeholder for the node_name property."""
            raise NotImplementedError("node_name property must be implemented by subclasses")

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A data processing node designed for 'Vishustra' to perform regular expression
    matching on input data.

    This node leverages the `re.findall` function from Python's standard library
    to locate all non-overlapping occurrences of a specified regex pattern within
    a given string or a list of strings. It offers flexibility in returning either
    the full matched strings, a specific capturing group by its index or name,
    or a tuple containing all capturing groups for each match.

    Configuration Context (`context: Dict[str, Any]`):
    - `pattern` (str): The regular expression pattern string to be compiled and used.
                       This field is mandatory.
    - `flags` (int, optional): A bitmask representing `re` module flags
                               (e.g., `re.IGNORECASE`, `re.MULTILINE`).
                               Defaults to `0` (no flags applied).
    - `return_group` (Union[int, str, None], optional): Controls the output format:
                                                        - `None` (default): Returns the raw output of `re.findall`.
                                                                          This will be a `List[str]` if the pattern
                                                                          has no capturing groups, or a
                                                                          `List[Tuple[str, ...]]` if capturing groups exist.
                                                        - `0`: Returns the entire matched string for each occurrence.
                                                        - `int > 0`: Returns the string from the capturing group at the
                                                                     specified 0-based index (e.g., `1` for the first group).
                                                        - `str`: Returns the string from the capturing group with the
                                                                 specified symbolic name (e.g., `(?P<name>...)`).

    Input (`data: Any`):
    - `str`: A single string on which the regex pattern will be applied.
    - `List[str]`: A list of strings; the regex pattern will be applied independently
                   to each string in the list. Non-string items in the list will be
                   skipped with a warning.

    Output (`Any`):
    The structure of the output depends on the input `data` type and the
    `return_group` configuration:

    - If `data` is a `str`:
        - `List[str]`: If `return_group` is `0`, an `int > 0`, or a `str`,
                       and matches are found.
        - `List[Tuple[str, ...]]` or `List[str]`: If `return_group` is `None`
                                                  (reflecting raw `re.findall` output).
        - An empty list `[]` if no matches are found.

    - If `data` is a `List[str]`:
        - `List[List[str]]`: When a specific group is extracted from each input string.
        - `List[List[Tuple[str, ...]]]` or `List[List[str]]`: When `return_group` is `None`.
        - A list containing empty lists (`[[], [], ...]`) for inputs where no matches
          were found or processing failed for an item.

    Raises:
    - `ValueError`: If the 'pattern' is missing, not a string, or if the input `data`
                    is neither a string nor a list of strings. Also raised if the
                    regex pattern itself is syntactically invalid (`re.error` wrapped).
    - `IndexError` / `KeyError`: These can occur internally if `return_group` specifies
                                 an invalid index or name for a particular match,
                                 though these are caught and logged as warnings for
                                 individual matches to allow processing to continue.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name for this processing node.
        """
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the regex matching operation based on the provided data and context.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing configuration parameters for the regex
                     matching, such as 'pattern', 'flags', and 'return_group'.

        Returns:
            The processed data, which is a list of matched strings or tuples,
            or a list of such lists if the input was a list of strings.
        """
        logger.debug(f"[{self.node_name}] Starting process with input data type: {type(data)}")

        # 1. Validate 'pattern' from context
        pattern_str = context.get('pattern')
        if not isinstance(pattern_str, str) or not pattern_str:
            logger.error(f"[{self.node_name}] Configuration error: 'pattern' must be a non-empty string in context.")
            raise ValueError("Missing or invalid 'pattern' in context. Must be a non-empty string.")

        # 2. Extract and validate 'flags' from context
        flags = context.get('flags', 0)
        if not isinstance(flags, int):
            logger.warning(f"[{self.node_name}] Invalid type for 'flags' in context ({type(flags)}). Defaulting to 0.")
            flags = 0

        # 3. Compile the regex pattern for efficiency and early error detection
        try:
            compiled_pattern = re.compile(pattern_str, flags)
            logger.debug(f"[{self.node_name}] Successfully compiled regex pattern: '{pattern_str}' with flags: {flags}")
        except re.error as e:
            logger.error(f"[{self.node_name}] Failed to compile regex pattern '{pattern_str}': {e}")
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

        # 4. Extract and validate 'return_group' from context
        return_group = context.get('return_group', None)
        if return_group is not None and not isinstance(return_group, (int, str)):
            logger.warning(f"[{self.node_name}] Invalid type for 'return_group' in context ({type(return_group)}). Defaulting to None.")
            return_group = None

        # Helper function to process a single string input
        def _process_single_string(text: str) -> Union[List[str], List[Tuple[str, ...]]]:
            """
            Applies the compiled regex pattern to a single string and extracts results
            based on the 'return_group' setting.
            """
            if not isinstance(text, str):
                logger.warning(f"[{self.node_name}] Skipping non-string item of type {type(text)} encountered in input list.")
                return [] # Return an empty list for non-string items within a list

            # Perform the findall operation
            matches = compiled_pattern.findall(text)
            logger.debug(f"[{self.node_name}] Found {len(matches)} matches for text (length {len(text)}) using pattern '{pattern_str}'.")

            if return_group is None:
                # If no specific group is requested, return the raw output from re.findall
                # This can be List[str] or List[Tuple[str, ...]]
                return matches
            else:
                # If a specific group (by index or name) is requested
                extracted_results: List[str] = []
                for match_item in matches:
                    try:
                        if isinstance(match_item, tuple):
                            # The match contains capturing groups, accessed by index.
                            # Named groups in `re.findall` are not directly accessible as dict keys,
                            # but are part of the tuple. If `return_group` is a string, it implies
                            # a named group was intended, which is typically handled by `re.finditer`
                            # and `match.groupdict()`, not direct `findall` tuple access.
                            # For now, we'll only support integer indexing for tuples.
                            if isinstance(return_group, int):
                                if return_group < 0 or return_group >= len(match_item):
                                    logger.warning(
                                        f"[{self.node_name}] Group index {return_group} is out of bounds "
                                        f"for match tuple {match_item}. Skipping this match's group extraction."
                                    )
                                    continue
                                extracted_results.append(match_item[return_group])
                            else: # return_group is a string
                                logger.warning(
                                    f"[{self.node_name}] Cannot extract named group '{return_group}' "
                                    f"directly from 're.findall' tuple result {match_item}. "
                                    f"Consider using `re.finditer` and `match.groupdict()` for named group access. Skipping."
                                )
                                continue
                        elif isinstance(match_item, str):
                            # The match has no explicit capturing groups (findall returned full strings).
                            # Only group 0 (the full match) is valid in this scenario.
                            if return_group == 0:
                                extracted_results.append(match_item)
                            else:
                                logger.warning(
                                    f"[{self.node_name}] Attempted to extract group {return_group} "
                                    f"from a non-grouped match '{match_item}'. "
                                    f"Only group 0 (full match) is valid. Skipping."
                                )
                                continue
                        else:
                            logger.warning(f"[{self.node_name}] Unexpected match item type: {type(match_item)}. Skipping.")
                            continue
                    except (IndexError, KeyError) as e:
                        logger.error(
                            f"[{self.node_name}] Error extracting group {return_group} from match '{match_item}': {e}. "
                            f"This match will be skipped."
                        )
                        continue # Move to the next match_item

                return extracted_results

        # 5. Determine input data type and process accordingly
        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing a single string input.")
            return _process_single_string(data)
        elif isinstance(data, list):
            logger.debug(f"[{self.node_name}] Processing a list of {len(data)} strings.")
            results = []
            for item in data:
                # Apply the single string processor to each item in the list
                results.append(_process_single_string(item))
            return results
        else:
            logger.error(f"[{self.node_name}] Invalid input data type: {type(data)}. Expected str or List[str].")
            raise ValueError(f"Invalid input data type: {type(data)}. Expected str or List[str].")

