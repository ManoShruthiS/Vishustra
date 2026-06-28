from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, List, Union, Optional
import re
import logging

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node that performs regular expression operations on input strings.
    This node supports various regex actions like matching, searching, finding all
    occurrences, substitution, and splitting.

    Configuration in the 'context' dictionary:
    - 'pattern' (str): The regular expression pattern to be used. (Required)
    - 'return_type' (str): Specifies the type of regex operation to perform.
      Supported values are:
        'match':   Equivalent to `re.match()`. Returns a `re.Match` object or `None`.
        'search':  Equivalent to `re.search()`. Returns a `re.Match` object or `None`.
        'findall': Equivalent to `re.findall()`. Returns a list of strings. (Default)
        'finditer': Equivalent to `re.finditer()`. Returns a list of `re.Match` objects.
        'sub':     Equivalent to `re.sub()`. Performs substitution. Requires 'repl'.
        'split':   Equivalent to `re.split()`. Splits the string by pattern.
      Defaults to 'findall' if not specified.
    - 'flags' (int, optional): Bitmask of `re` flags (e.g., `re.IGNORECASE`, `re.MULTILINE`).
      Defaults to 0 (no flags).
    - 'repl' (str, optional): The replacement string for the 'sub' operation.
      This parameter is mandatory when 'return_type' is 'sub'.
    - 'count' (int, optional): The maximum number of substitutions for 'sub' operation.
      Defaults to 0 (all occurrences).
    - 'maxsplit' (int, optional): The maximum number of splits for 'split' operation.
      Defaults to 0 (all occurrences).

    Input `data` can be a single string or a list of strings.
    If `data` is a list, the regex operation is applied to each string individually,
    and a list of results is returned.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying a configured regular expression operation.

        Args:
            data: The input string or a list of strings to be processed.
            context: A dictionary containing the configuration parameters for
                     the regex operation (e.g., 'pattern', 'return_type', 'flags').

        Returns:
            The result of the regex operation. The type of the result varies based on
            the 'return_type' configured in the context. If the input `data` was
            a list, a list of individual results is returned. Returns `None` for
            skipped or errored items in a list, or if a single input resulted in no
            match/find and `return_type` allows `None` (e.g., 'match', 'search').

        Raises:
            ValueError: If the 'pattern' is missing, invalid, or 'return_type' is
                        unsupported, or 'repl' is missing for 'sub' operation.
            TypeError: If the input `data` is not a string or a list of strings.
        """
        pattern_str = context.get('pattern')
        if not isinstance(pattern_str, str) or not pattern_str:
            logger.error("RegexMatcherNode: 'pattern' (string) is a mandatory context parameter.")
            raise ValueError("Missing or invalid 'pattern' in context for RegexMatcherNode.")

        return_type = context.get('return_type', 'findall').lower()
        flags = context.get('flags', 0)
        repl = context.get('repl')
        count = context.get('count', 0)
        maxsplit = context.get('maxsplit', 0)

        try:
            compiled_pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.error(f"RegexMatcherNode: Invalid regex pattern '{pattern_str}': {e}")
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

        if not isinstance(data, (str, list)):
            logger.error(
                f"RegexMatcherNode: Received unsupported data type: {type(data)}. "
                "Expected str or list[str]."
            )
            raise TypeError(
                f"RegexMatcherNode expects input data to be a string or a list of strings, "
                f"but received {type(data)}."
            )

        is_list_input = isinstance(data, list)
        items_to_process = data if is_list_input else [data]
        results = []

        for item_idx, item in enumerate(items_to_process):
            if not isinstance(item, str):
                logger.warning(
                    f"RegexMatcherNode: Skipping non-string item at index {item_idx} "
                    f"(type: {type(item)}) in input list. Expected str."
                )
                results.append(None)
                continue

            try:
                if return_type == 'match':
                    result = compiled_pattern.match(item)
                elif return_type == 'search':
                    result = compiled_pattern.search(item)
                elif return_type == 'findall':
                    result = compiled_pattern.findall(item)
                elif return_type == 'finditer':
                    result = list(compiled_pattern.finditer(item)) # Convert iterator to list for concrete output
                elif return_type == 'sub':
                    if repl is None:
                        logger.error(
                            "RegexMatcherNode: 'repl' (string) is a mandatory context parameter "
                            "when 'return_type' is 'sub'."
                        )
                        raise ValueError(
                            "'repl' must be provided in context for 'sub' operation in RegexMatcherNode."
                        )
                    result = compiled_pattern.sub(repl, item, count)
                elif return_type == 'split':
                    result = compiled_pattern.split(item, maxsplit)
                else:
                    logger.error(
                        f"RegexMatcherNode: Unsupported 'return_type' '{return_type}' provided in context. "
                        "Must be one of 'match', 'search', 'findall', 'finditer', 'sub', 'split'."
                    )
                    raise ValueError(
                        f"Unsupported 'return_type': '{return_type}'. "
                        "Refer to documentation for supported types."
                    )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"RegexMatcherNode: Error performing '{return_type}' operation on item "
                    f"'{item[:100]}...' at index {item_idx}: {e}", exc_info=True
                )
                results.append(None) # Append None if an error occurs for this specific item

        # If original input was a single string, return its result directly.
        # Otherwise, return the list of results.
        if is_list_input:
            return results
        else:
            return results[0] if results else None # Handle case where items_to_process was empty or only non-strings.