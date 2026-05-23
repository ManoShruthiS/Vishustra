import re
import logging
from typing import Any, Dict, List, Set

# Assuming BaseNode is located at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract unique URLs from various
    text-based input data types.

    This node leverages a regular expression to identify common URL patterns,
    including those starting with 'http://', 'https://', or 'www.'. It
    supports processing individual strings or lists of strings, ensuring
    all extracted URLs are unique in the final output.
    """

    # A compiled regular expression for robust URL detection.
    # It targets common URL structures:
    # - Starts with http://, https://, or www.
    # - Followed by valid domain characters (letters, numbers, hyphens, dots)
    # - Ends with a TLD of 2 to 6 characters
    # - Optionally includes a path, query, or fragment (any non-whitespace characters)
    # Word boundaries (\b) are used to prevent partial matches within words.
    _URL_REGEX = re.compile(
        r'\b(?:https?://|www\.)(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}(?:/[^\s]*)?\b'
    )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the provided input data.

        This method supports two primary data types for extraction:
        - A single string: All URLs within the string will be extracted.
        - A list of strings: URLs will be extracted from each string in the list.
          Non-string items within the list will be gracefully skipped with a debug log.

        For any other unsupported data type, a warning will be logged, and an
        empty list will be returned, indicating no URLs could be processed.

        Args:
            data (Any): The input data expected to be a `str` or `List[str]`.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. While not
                                       directly utilized by this specific node, it
                                       adheres to the `BaseNode` interface.

        Returns:
            List[str]: A list of unique URLs found in the input data, sorted
                       alphabetically for consistent output. Returns an empty
                       list if no URLs are found or if the input data type is
                       unsupported.
        """
        unique_urls: Set[str] = set()

        if isinstance(data, str):
            # Process a single string input
            extracted_from_string = self._URL_REGEX.findall(data)
            unique_urls.update(extracted_from_string)
            logger.debug(f"Extracted {len(extracted_from_string)} URLs from input string.")

        elif isinstance(data, list):
            # Process each item in a list, expecting strings
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    extracted_from_item = self._URL_REGEX.findall(item)
                    unique_urls.update(extracted_from_item)
                    logger.debug(f"Extracted {len(extracted_from_item)} URLs from list item at index {idx}.")
                else:
                    logger.debug(
                        f"Skipping non-string item at index {idx} in input list "
                        f"(type: {type(item).__name__}). URLExtractor expects strings."
                    )
        else:
            # Handle unsupported data types
            logger.warning(
                f"URLExtractorNode received unsupported data type: {type(data).__name__}. "
                f"Expected 'str' or 'List[str]'. Returning an empty list."
            )
            return []  # Immediately return an empty list for unsupported types

        # Convert the set of unique URLs to a sorted list for consistent output
        result_list = sorted(list(unique_urls))
        logger.info(f"Successfully identified and extracted {len(result_list)} unique URLs.")
        return result_list