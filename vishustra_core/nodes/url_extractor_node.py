import re
import logging
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node responsible for extracting URLs from textual content.

    This node efficiently identifies and extracts unique URLs from a single string
    or a list of strings, providing robust handling for various input formats.
    """

    # Pre-compile a regular expression for URL extraction for optimal performance.
    # This regex is designed to capture common URL patterns, including 'http://' and 'https://'
    # schemes, and a broad range of valid URL characters.
    _URL_REGEX = re.compile(
        r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all unique URLs.

        The `data` input can be a single string or a list of strings.
        - If `data` is a string, URLs are extracted directly from it.
        - If `data` is a list of strings, URLs are extracted from each string in the list.
        - Non-string items within a list are gracefully skipped with a warning.
        - Other unsupported data types are logged as warnings, and an empty list is returned.

        Args:
            data: The input content from which to extract URLs. Expected to be `str` or `List[str]`.
            context: A dictionary holding contextual information for the processing pipeline.
                     This node does not currently utilize the `context`.

        Returns:
            A sorted list of unique URLs found in the input data. An empty list is returned
            if no URLs are found, or if the input data format is unsupported.
        """
        extracted_urls: Set[str] = set()

        if data is None:
            logger.warning(
                "Received 'None' as input data for URLExtractorNode. "
                "Returning an empty list as no content to process."
            )
            return []

        if isinstance(data, str):
            found_urls = self._URL_REGEX.findall(data)
            extracted_urls.update(found_urls)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    found_urls = self._URL_REGEX.findall(item)
                    extracted_urls.update(found_urls)
                else:
                    # Log a warning for non-string elements within the list
                    logger.warning(
                        f"Skipping an item of type '{type(item).__name__}' in the input list for "
                        "URLExtractorNode. Expected string elements for URL extraction."
                    )
        else:
            # Log a warning for completely unsupported data types
            logger.warning(
                f"Unsupported data type received for URLExtractorNode: '{type(data).__name__}'. "
                "Expected 'str' or 'List[str]'. Returning an empty list."
            )
            return []

        # Convert the set of unique URLs to a sorted list for consistent and deterministic output.
        return sorted(list(extracted_urls))