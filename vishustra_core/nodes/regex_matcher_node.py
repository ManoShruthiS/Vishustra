import logging
import re
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra node that performs regular expression matching and extraction
    from input data.

    This node expects the input `data` to be a string. It uses a regex pattern
    provided in the `context` dictionary.

    Configuration in `context`:
    - `regex_pattern` (str): The regular expression pattern to use for matching. (Required)
    - `return_all_matches` (bool): If True, `process` returns a list of all
      non-overlapping matches found. If False (default), it returns only the
      first full match string, or `None` if no match is found.

    Returns from `process`:
    - If `return_all_matches` is True: `List[str]` containing all matches.
    - If `return_all_matches` is False: `str` (the first full match) or `None`
      (if no match is found).
    - Returns `None` if the input `data` is not a string, `regex_pattern` is
      missing or invalid in `context`, or a `re.error` occurs during pattern compilation/matching.
      Errors are logged appropriately.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], str, None]:
        """
        Processes the input data using a regular expression pattern to find matches.

        Args:
            data: The input data, expected to be a string within which to search.
            context: A dictionary containing operational parameters for the node.
                     Must include 'regex_pattern' (str). Can optionally include
                     'return_all_matches' (bool, defaults to False).

        Returns:
            A list of all found matches (if `return_all_matches` is True),
            the first full match string (if `return_all_matches` is False and a match is found),
            or `None` (if no match, or an error occurred during processing).
        """
        if not isinstance(data, str):
            logger.error(
                "[{}] Invalid input data type. Expected string, got '{}' for data: {}".format(
                    self.node_name, type(data).__name__, data
                )
            )
            return None

        regex_pattern = context.get("regex_pattern")
        if not isinstance(regex_pattern, str) or not regex_pattern:
            logger.error(
                "[{}] Missing or invalid 'regex_pattern' in context. Expected a non-empty string.".format(
                    self.node_name
                )
            )
            return None

        return_all_matches = context.get("return_all_matches", False)
        if not isinstance(return_all_matches, bool):
            logger.warning(
                "[{}] Invalid type for 'return_all_matches' in context. Expected boolean, got '{}'. Defaulting to False.".format(
                    self.node_name, type(return_all_matches).__name__
                )
            )
            return_all_matches = False

        try:
            if return_all_matches:
                matches = re.findall(regex_pattern, data)
                logger.debug(
                    "[{}] Found {} matches for pattern '{}' in data (first 50 chars): '{}'".format(
                        self.node_name, len(matches), regex_pattern, data[:50]
                    )
                )
                return matches
            else:
                match = re.search(regex_pattern, data)
                if match:
                    logger.debug(
                        "[{}] Found first match '{}' for pattern '{}' in data (first 50 chars): '{}'".format(
                            self.node_name, match.group(0), regex_pattern, data[:50]
                        )
                    )
                    return match.group(0)
                else:
                    logger.debug(
                        "[{}] No match found for pattern '{}' in data (first 50 chars): '{}'".format(
                            self.node_name, regex_pattern, data[:50]
                        )
                    )
                    return None
        except re.error as e:
            logger.error(
                "[{}] Regular expression error with pattern '{}': {}".format(
                    self.node_name, regex_pattern, e
                )
            )
            return None
        except Exception as e:
            logger.error(
                "[{}] An unexpected error occurred during regex processing: {}".format(
                    self.node_name, e
                )
            )
            return None