import re
import logging
from typing import Any, Dict, List, Union, Iterable

# Assuming BaseNode is available at this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression pattern
    to input data to find and extract matches.

    The node expects the regex pattern to be provided in the context under
    the key 'regex_pattern'.

    If the input data is a single string, it returns a list of all non-overlapping
    matches found by the pattern.
    If the input data is an iterable of strings, it processes each string
    individually and returns a list of lists of matches.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data using a regular expression defined in the context.

        Args:
            data: The input data, expected to be a string or an iterable of strings.
            context: A dictionary containing execution context, expected to have:
                     - 'regex_pattern' (str): The regular expression pattern to use.

        Returns:
            A list of strings if the input data is a single string,
            or a list of lists of strings if the input data is an iterable of strings.
            Returns an empty list or list of empty lists if no matches are found
            or if the input data is invalid.

        Raises:
            ValueError: If 'regex_pattern' is missing from the context.
            re.error: If the provided regex pattern is syntactically invalid.
            TypeError: If the input data is not a string or an iterable of strings
                       or if iterable contains bytes/bytearray.
        """
        regex_pattern = context.get("regex_pattern")
        if not regex_pattern:
            logger.error(
                "[%s] 'regex_pattern' is missing from the context. Cannot perform regex match.",
                self.node_name
            )
            raise ValueError(
                f"Context missing required key 'regex_pattern' for {self.node_name}."
            )

        try:
            compiled_pattern = re.compile(regex_pattern)
        except re.error as e:
            logger.error(
                "[%s] Invalid regex pattern '%s' provided in context: %s",
                self.node_name, regex_pattern, e
            )
            raise re.error(f"Invalid regex pattern provided: {regex_pattern} - {e}")

        if isinstance(data, str):
            try:
                matches = compiled_pattern.findall(data)
                logger.debug(
                    "[%s] Processed single string data. Found %d matches.",
                    self.node_name, len(matches)
                )
                return matches
            except TypeError as e:
                # This specific TypeError might indicate issues within re.findall with data type.
                logger.error(
                    "[%s] Data provided to regex matching was of incompatible type for string processing: %s",
                    self.node_name, type(data)
                )
                raise TypeError(
                    f"RegexMatcherNode received data of incompatible type for string processing: {type(data)}"
                )
        elif isinstance(data, Iterable):
            # Explicitly disallow byte strings which are also iterable but not strings
            if isinstance(data, (bytes, bytearray)):
                logger.error(
                    "[%s] Data is expected to be a string or iterable of strings, but received bytes/bytearray.",
                    self.node_name
                )
                raise TypeError(
                    f"RegexMatcherNode received bytes/bytearray. Expected string or iterable of strings."
                )

            results: List[List[str]] = []
            for item in data:
                if not isinstance(item, str):
                    logger.warning(
                        "[%s] Skipping non-string item in iterable data: %s (type: %s). Appending empty list.",
                        self.node_name, item, type(item)
                    )
                    results.append([])
                    continue
                matches_for_item = compiled_pattern.findall(item)
                results.append(matches_for_item)
            logger.debug(
                "[%s] Processed iterable data. Generated %d result sets.",
                self.node_name, len(results)
            )
            return results
        else:
            logger.error(
                "[%s] Input data must be a string or an iterable of strings, but received type: %s",
                self.node_name, type(data)
            )
            raise TypeError(
                f"RegexMatcherNode expects data to be a string or an iterable of strings, "
                f"but received {type(data)}."
            )