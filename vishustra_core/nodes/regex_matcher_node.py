import re
import logging
from typing import Any, Dict, Union, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node that performs regular expression matching on input data.

    This node extracts all non-overlapping matches of a specified regex pattern
    from a string or a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], List[List[str]]]:
        """
        Processes the input data by applying a regular expression pattern.

        The regex pattern must be provided in the `context` dictionary under the key 'pattern'.
        Optional regex flags (e.g., re.IGNORECASE, re.MULTILINE) can be provided
        under the key 'flags'.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing execution context,
                     must include 'pattern' (str) and can include 'flags' (int, e.g., re.IGNORECASE).

        Returns:
            A list of strings if the input `data` was a single string,
            or a list of lists of strings if the input `data` was a list of strings.
            Each inner list contains all non-overlapping matches found for that string.

        Raises:
            ValueError: If 'pattern' is missing from the context.
            TypeError: If the input data is not a string or a list of strings.
            re.error: If the provided regex pattern is invalid.
        """
        if 'pattern' not in context:
            error_msg = f"'{self.node_name}' requires a 'pattern' key in the context."
            logger.error(error_msg)
            raise ValueError(error_msg)

        pattern = context['pattern']
        flags = context.get('flags', 0)

        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            error_msg = f"Invalid regex pattern '{pattern}' provided to '{self.node_name}': {e}"
            logger.error(error_msg)
            raise re.error(error_msg) from e

        if isinstance(data, str):
            matches = compiled_pattern.findall(data)
            logger.debug(f"'{self.node_name}' found {len(matches)} matches in single string data.")
            return matches
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                error_msg = (f"'{self.node_name}' received a list containing non-string elements. "
                             f"Expected list of strings.")
                logger.error(error_msg)
                raise TypeError(error_msg)

            results: List[List[str]] = []
            for item in data:
                matches = compiled_pattern.findall(item)
                results.append(matches)
            logger.debug(f"'{self.node_name}' processed {len(data)} strings, finding matches in each.")
            return results
        else:
            error_msg = (f"'{self.node_name}' received unsupported data type: {type(data)}. "
                         f"Expected str or List[str].")
            logger.error(error_msg)
            raise TypeError(error_msg)