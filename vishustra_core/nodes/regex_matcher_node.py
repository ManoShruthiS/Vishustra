import re
import logging
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node that performs regular expression matching on input data.

    It extracts substrings from the input data based on a provided regex pattern
    and optional flags, returning either the first match or a list of all matches.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[str, List[str], None]:
        """
        Processes the input data using a regular expression pattern specified in the context.

        The `data` input is expected to be a string.
        The `context` dictionary must contain:
        - 'regex_pattern' (str): The regular expression pattern to use.

        Optional `context` parameters:
        - 'regex_flags' (int): Bitmask of regex flags (e.g., re.IGNORECASE, re.MULTILINE). Defaults to 0.
        - 'return_first_match_only' (bool): If True, returns only the first matched string.
                                           If False, returns a list of all matched strings. Defaults to False.

        Args:
            data (Any): The string data to apply the regex pattern against.
            context (Dict[str, Any]): A dictionary containing parameters for the regex matching.

        Returns:
            Union[str, List[str], None]:
                - If 'return_first_match_only' is True: The first matched string or None if no match.
                - If 'return_first_match_only' is False: A list of all matched strings (can be empty).

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'regex_pattern' is missing from the context or is invalid.
        """
        if not isinstance(data, str):
            logger.error(f"Node '{self.node_name}': Input data must be a string, but received type {type(data)}.")
            raise TypeError(f"Input data for '{self.node_name}' must be a string.")

        regex_pattern = context.get('regex_pattern')
        if not isinstance(regex_pattern, str):
            logger.error(f"Node '{self.node_name}': 'regex_pattern' missing or not a string in context.")
            raise ValueError(f"'regex_pattern' (string) is required in context for '{self.node_name}'.")

        regex_flags = context.get('regex_flags', 0)
        return_first_match_only = context.get('return_first_match_only', False)

        logger.debug(f"Node '{self.node_name}': Applying pattern '{regex_pattern}' with flags {regex_flags} "
                     f"to data (first 50 chars): '{data[:50]}...'")

        try:
            compiled_regex = re.compile(regex_pattern, regex_flags)
            if return_first_match_only:
                match = compiled_regex.search(data)
                result = match.group(0) if match else None
                logger.debug(f"Node '{self.node_name}': First match result: {result}")
            else:
                result = compiled_regex.findall(data)
                logger.debug(f"Node '{self.node_name}': All matches result ({len(result)} found): {result}")
            return result
        except re.error as e:
            logger.error(f"Node '{self.node_name}': Invalid regex pattern '{regex_pattern}': {e}")
            raise ValueError(f"Invalid regex pattern provided to '{self.node_name}': {e}")
        except Exception as e:
            logger.exception(f"Node '{self.node_name}': An unexpected error occurred during processing.")
            raise