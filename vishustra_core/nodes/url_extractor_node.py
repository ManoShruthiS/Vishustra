import re
import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from text data.

    This node can process either a single string or an iterable of strings,
    identifying common URL patterns and returning all unique URLs found.
    """

    # Regex pattern for identifying URLs.
    # It covers common http(s):// and www. prefixes, followed by one or more
    # non-whitespace characters. This balances robustness with avoiding overly
    # complex and potentially slow regex patterns for general extraction.
    _URL_PATTERN = re.compile(r'\b(?:https?://|www\.)\S+\b')

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Helper method to extract URLs from a single string using the compiled regex.
        """
        try:
            return self._URL_PATTERN.findall(text)
        except Exception as e:
            logger.error(f"Failed to apply URL regex pattern to text due to: {e}", exc_info=True)
            return []

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        The method supports the following input data types:
        - `str`: Extracts URLs directly from the string.
        - `list` or `tuple` of `str`: Iterates through the collection,
          extracting URLs from each string element. Non-string elements
          within the collection are logged as warnings and skipped.
        - Any other type: Logs a warning and returns an empty list.

        Args:
            data: The input data, expected to be a string or an iterable
                  of strings, from which URLs will be extracted.
            context: A dictionary containing contextual information for the node's
                     operation, such as a request ID. (Used for logging, not core logic).

        Returns:
            A sorted list of unique URLs found within the input data.
        """
        extracted_urls: List[str] = []
        request_id = context.get('request_id', 'N/A')

        if isinstance(data, str):
            logger.debug(f"URLExtractorNode processing single string. Request ID: {request_id}")
            extracted_urls.extend(self._extract_urls_from_text(data))
        elif isinstance(data, (list, tuple)):
            logger.debug(f"URLExtractorNode processing list/tuple of strings. Request ID: {request_id}")
            for i, item in enumerate(data):
                if isinstance(item, str):
                    extracted_urls.extend(self._extract_urls_from_text(item))
                else:
                    logger.warning(
                        f"URLExtractorNode skipped non-string item at index {i} "
                        f"(type: {type(item)}) in input collection. Expected string for URL extraction. "
                        f"Request ID: {request_id}"
                    )
        else:
            logger.warning(
                f"URLExtractorNode received unsupported data type: {type(data)}. "
                f"Expected `str` or `List[str]`. Returning empty list. Request ID: {request_id}"
            )
            return []

        # Convert to a set to ensure uniqueness, then back to a sorted list.
        # Sorting provides consistent output for identical inputs.
        return sorted(list(set(extracted_urls)))