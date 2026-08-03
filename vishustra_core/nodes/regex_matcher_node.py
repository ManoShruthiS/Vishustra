import re
import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is available via this import path as specified by project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regex matching on input data.

    It extracts all non-overlapping occurrences of a specified regex pattern
    from the input string data. The node is robust to invalid patterns and
    non-string inputs, raising specific errors.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data by applying a regular expression pattern
        and returning all found matches.

        The `context` dictionary is expected to contain:
        - 'pattern' (str): The regular expression pattern string to match.

        Optional `context` parameters:
        - 'flags' (int): Bitmask of regex flags (e.g., re.IGNORECASE, re.MULTILINE).
                         Defaults to 0 (no flags) if not provided.

        Args:
            data (Any): The input data. It will be converted to a string before
                        regex application.
            context (Dict[str, Any]): A dictionary containing node-specific
                                       configuration, primarily the regex 'pattern'
                                       and optional 'flags'.

        Returns:
            List[str]: A list of all non-overlapping string matches found by
                       `re.findall`. Returns an empty list if no matches are found.

        Raises:
            ValueError: If 'pattern' is missing from the context or is not a string.
            TypeError: If the input 'data' cannot be converted to a string.
            re.error: If the provided regex 'pattern' is syntactically invalid.
            Exception: For any other unexpected errors during processing.
        """
        # Validate 'pattern' in context
        if 'pattern' not in context or not isinstance(context['pattern'], str):
            logger.error("RegexMatcherNode: Missing or invalid 'pattern' in context. Expected a string.")
            raise ValueError("Context must contain a string 'pattern' for regex matching.")

        pattern_str: str = context['pattern']
        flags: int = context.get('flags', 0)

        # Convert data to string
        try:
            input_string: str = str(data)
        except Exception as e:
            logger.error(
                f"RegexMatcherNode: Failed to convert input data of type "
                f"'{type(data).__name__}' to string. Error: {e}",
                exc_info=True
            )
            raise TypeError(f"Input data must be convertible to string. Got type {type(data).__name__}.") from e

        # Perform regex matching
        try:
            compiled_pattern = re.compile(pattern_str, flags)
            matches: List[str] = compiled_pattern.findall(input_string)
            logger.debug(f"RegexMatcherNode: Successfully found {len(matches)} matches for pattern '{pattern_str}'.")
            return matches
        except re.error as e:
            logger.error(
                f"RegexMatcherNode: Invalid regex pattern '{pattern_str}' provided. Error: {e}",
                exc_info=True
            )
            raise re.error(f"Invalid regex pattern provided: {e}") from e
        except Exception as e:
            logger.error(
                f"RegexMatcherNode: An unexpected error occurred during regex matching with pattern '{pattern_str}'. Error: {e}",
                exc_info=True
            )
            raise # Re-raise unexpected exceptions for upstream handling
