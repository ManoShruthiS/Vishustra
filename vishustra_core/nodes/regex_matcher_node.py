import re
import logging
from typing import Any, Dict, List, Union, Iterable

# Assuming 'vishustra_core.nodes.base_node' is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node that performs regex matching on input data.

    This node can process a single string or an iterable of strings,
    applying a regex pattern provided in the context and returning all
    non-overlapping matches.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Union[str, Iterable[str]], context: Dict[str, Any]) -> Union[List[str], Dict[str, List[str]]]:
        """
        Processes the input data using a regex pattern from the context.

        The method expects a 'pattern' key in the context dictionary, whose
        value is the regex string to be applied.

        Args:
            data: The input data, which can be a single string or an iterable
                  (e.g., list, tuple) of strings.
            context: A dictionary containing node-specific parameters.
                     Expected key: 'pattern' (str) - the regex pattern to match.

        Returns:
            If `data` is a single string, a `List[str]` containing all
            non-overlapping matches found.
            If `data` is an iterable of strings, a `Dict[str, List[str]]` where
            keys are the original input strings and values are lists of their
            respective matches.

        Raises:
            ValueError: If the 'pattern' key is missing, empty, or not a string
                        in the `context`.
            TypeError: If the input `data` is not a string or an iterable of strings.
                       Also, if an item within an iterable `data` is not a string.
            re.error: If the provided regex pattern is syntactically invalid.
        """
        pattern_str = context.get('pattern')

        if not isinstance(pattern_str, str) or not pattern_str.strip():
            logger.error("RegexMatcherNode: 'pattern' must be a non-empty string in the context.")
            raise ValueError("Missing or invalid 'pattern' in context. A non-empty string is required.")

        try:
            # Compile the regex pattern for efficiency, especially with repeated use
            compiled_pattern = re.compile(pattern_str)
            logger.debug(f"RegexMatcherNode: Successfully compiled regex pattern: '{pattern_str}'")
        except re.error as e:
            logger.error(f"RegexMatcherNode: Invalid regex pattern '{pattern_str}': {e}")
            raise  # Re-raise the original regex error

        if isinstance(data, str):
            logger.debug(f"RegexMatcherNode: Applying pattern to single string input.")
            return compiled_pattern.findall(data)
        elif isinstance(data, Iterable):
            results: Dict[str, List[str]] = {}
            for item in data:
                if isinstance(item, str):
                    logger.debug(f"RegexMatcherNode: Applying pattern to iterable item: '{item[:50]}...'")
                    results[item] = compiled_pattern.findall(item)
                else:
                    # Log a warning and skip non-string items to be robust
                    logger.warning(f"RegexMatcherNode: Skipping non-string item in iterable input: type={type(item)}. "
                                   "Only string elements are processed.")
            return results
        else:
            logger.error(f"RegexMatcherNode: Input data must be a string or an iterable of strings, "
                         f"but received type: {type(data)}.")
            raise TypeError("Input data must be a string or an iterable of strings.")