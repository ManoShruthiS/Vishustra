import re
import logging
from typing import Any, Dict, List, Union

# Assuming the BaseNode class is located at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

# Set up logging for this module
logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that applies a regular expression pattern
    to the input data and extracts all non-overlapping matches.

    This node is designed to process either a single string or a list of strings.
    It returns all found matches according to the configured regex pattern.
    """

    def __init__(self, pattern: str):
        """
        Initializes the RegexMatcherNode with a regular expression pattern.

        The pattern is compiled during initialization for efficiency in subsequent
        `process` calls.

        Args:
            pattern: The regular expression string to be used for matching.
                     This pattern must be a valid regex string.

        Raises:
            TypeError: If the provided pattern is not a string.
            re.error: If the provided pattern is an invalid regular expression
                      that cannot be compiled.
        """
        if not isinstance(pattern, str):
            logger.error("Initialization failed: pattern must be a string. Received type: %s", type(pattern))
            raise TypeError(f"Pattern for RegexMatcherNode must be a string, got {type(pattern)}.")
        try:
            self._compiled_pattern = re.compile(pattern)
            logger.debug("RegexMatcherNode initialized successfully with pattern: '%s'", pattern)
        except re.error as e:
            logger.error("Initialization failed: invalid regex pattern '%s'. Error: %s", pattern, e)
            raise re.error(f"Failed to compile regex pattern '{pattern}': {e}") from e

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[str], List[List[str]]]:
        """
        Processes the input data by applying the node's configured regex pattern
        to find all non-overlapping matches.

        The behavior depends on the type of `data` provided:
        - If `data` is a `str`, it returns a `List[str]` containing all matches found
          within that string.
        - If `data` is a `List[str]`, it iterates through each string in the list,
          applies the regex, and returns a `List[List[str]]` where each inner list
          contains matches for the corresponding input string. Non-string elements
          within the list will result in an empty list for that position after a warning.

        Args:
            data: The input data to be processed. Expected to be either a `str`
                  or a `List[str]`.
            context: A dictionary containing additional context relevant to the
                     orchestration. This node does not currently utilize the context
                     but adheres to the `BaseNode` interface.

        Returns:
            A list of strings (if input `data` was a single string) or a list of lists
            of strings (if input `data` was a list of strings). Each list contains
            all non-overlapping matches found by the regex. Returns an empty list
            or list of empty lists if no matches are found or if an unsupported
            data type is encountered (after raising an error for top-level type).

        Raises:
            TypeError: If the top-level `data` provided is neither a string nor a
                       list of strings.
        """
        logger.info("RegexMatcherNode received data of type: %s for processing.", type(data))

        if isinstance(data, str):
            matches = self._compiled_pattern.findall(data)
            logger.debug("Processed single string input. Found %d matches.", len(matches))
            return matches
        elif isinstance(data, list):
            results: List[List[str]] = []
            for i, item in enumerate(data):
                if isinstance(item, str):
                    item_matches = self._compiled_pattern.findall(item)
                    results.append(item_matches)
                    logger.debug("Processed list item %d (string). Found %d matches.", i, len(item_matches))
                else:
                    logger.warning(
                        "RegexMatcherNode encountered a non-string item at index %d in the input list. "
                        "Expected str, but got %s. Skipping this item and returning an empty list for it.",
                        i, type(item)
                    )
                    results.append([]) # Append an empty list for non-string items within the list
            logger.info("Finished processing list of strings. %d items processed.", len(data))
            return results
        else:
            logger.error(
                "RegexMatcherNode received unsupported data type. Expected 'str' or 'List[str]', "
                "but got '%s'.", type(data)
            )
            raise TypeError(
                f"Unsupported data type for RegexMatcherNode. Expected `str` or `List[str]`, "
                f"but received `{type(data)}`."
            )