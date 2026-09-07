import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from various data types.

    This node is designed to identify and collect URLs embedded within input data,
    supporting common string-based formats.

    It can process:
    - A single string: Extracts all URLs found within the string.
    - A list or tuple of strings: Iterates through each string and extracts URLs.
    - A dictionary: Iterates through string values within the dictionary and extracts URLs.
    - Other types: Logs a warning and returns an empty list, ensuring robustness.

    URLs are identified using a practical regular expression that covers both
    `http(s)://` and `www.` prefixes, with a basic cleanup for common trailing punctuation.
    """

    # A compiled regular expression for practical URL detection in text.
    # This pattern aims to capture URLs starting with `http://`, `https://`, or `www.`,
    # stopping at common delimiters like whitespace, angle brackets (`<`, `>`), or double quotes (`"`).
    _URL_PATTERN = re.compile(
        r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def _extract_from_text(self, text: str) -> List[str]:
        """
        Helper method to extract URLs from a single string.
        Performs basic cleanup to remove common trailing punctuation not typically part of a URL.

        Args:
            text (str): The input string to search for URLs.

        Returns:
            List[str]: A list of cleaned URLs found in the text.
        """
        found_urls = self._URL_PATTERN.findall(text)
        cleaned_urls = []
        for url in found_urls:
            # Basic heuristic cleanup: remove common trailing punctuation
            # if it appears at the end of the extracted URL.
            # This helps in cases like "visit example.com." to correctly extract "example.com".
            while url and url[-1] in '.,?!;':
                url = url[:-1]
            if url:  # Ensure the URL isn't empty after stripping
                cleaned_urls.append(url)
        return cleaned_urls

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract unique URLs.

        The method gracefully handles various input data types, including single strings,
        lists/tuples of strings, and dictionaries with string values. Non-string components
        within collections are skipped with a warning.

        Args:
            data (Any): The input data. Expected types are `str`, `list[str]`, `tuple[str]`,
                        or `dict[str, str]`.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       This node does not directly utilize the context
                                       but adheres to the `BaseNode` interface.

        Returns:
            List[str]: A sorted list of unique URLs found in the data. Returns an empty list
                       if no URLs are found, if the data type is unsupported, or if an
                       unexpected error occurs during processing.
        """
        extracted_urls = set()
        
        try:
            if isinstance(data, str):
                logger.debug(f"Node '{self.node_name}': Processing single string data for URLs.")
                extracted_urls.update(self._extract_from_text(data))
            elif isinstance(data, (list, tuple)):
                logger.debug(f"Node '{self.node_name}': Processing list/tuple data for URLs.")
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        extracted_urls.update(self._extract_from_text(item))
                    else:
                        logger.warning(
                            f"Node '{self.node_name}': Skipping non-string item at index {i} "
                            f"in input collection. Type: {type(item).__name__}."
                        )
            elif isinstance(data, dict):
                logger.debug(f"Node '{self.node_name}': Processing dictionary data for URLs.")
                for key, value in data.items():
                    if isinstance(value, str):
                        extracted_urls.update(self._extract_from_text(value))
                    else:
                        logger.warning(
                            f"Node '{self.node_name}': Skipping non-string value for key '{key}' "
                            f"in input dictionary. Type: {type(value).__name__}."
                        )
            else:
                logger.warning(
                    f"Node '{self.node_name}': Unsupported data type for URL extraction. "
                    f"Expected `str`, `list[str]`, `tuple[str]`, or `dict[str, str]`, "
                    f"but received type: {type(data).__name__}. Returning an empty list."
                )
        except Exception as e:
            logger.error(
                f"Node '{self.node_name}': An unexpected error occurred during URL extraction: {e}",
                exc_info=True  # Logs the full traceback for debugging
            )
            # In a resilient orchestration framework, returning an empty list on error
            # often allows the pipeline to continue, while the error is logged for investigation.
            return [] 

        # Return a sorted list of unique URLs for consistent output.
        return sorted(list(extracted_urls))