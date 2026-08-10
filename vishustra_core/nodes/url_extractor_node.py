import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A processing node designed to extract URLs from text data.

    This node leverages a robust regular expression to identify common URL patterns,
    including those starting with 'http://', 'https://', and 'www.'. It can process
    either a single string or a list of strings, providing a list of all discovered URLs.
    Non-string elements within a list input are gracefully skipped with a warning.
    """

    # A comprehensive regex pattern to capture various URL formats.
    # It accounts for schemes (http/https), optional 'www.', domain names,
    # port numbers, paths, query parameters, and fragments.
    # This pattern is designed to be largely compliant with RFC 3986 (URI Generic Syntax)
    # and common web practices, while also being practical for text extraction.
    _URL_PATTERN = re.compile(
        r'(?:https?://|www\.)'  # Scheme or 'www.' prefix
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain name and subdomains
        r'[a-zA-Z]{2,6}'  # TLD (e.g., .com, .org, .co.uk)
        r'(?::\d{2,5})?'  # Optional port number
        r'(?:/?|[/?]\S+)'  # Optional path, query, or fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract URLs.

        The node expects input data to be either a string or a list of strings.
        If a list is provided, each string element within the list will be processed.
        Non-string elements in a list are logged as warnings and skipped.
        Unsupported data types for the main input 'data' will also trigger a warning
        and result in an empty list being returned.

        Args:
            data: The input content to be scanned for URLs. Expected types are `str`
                  or `list[str]`.
            context: A dictionary providing contextual information for the processing
                     (not utilized by this specific node).

        Returns:
            A `list[str]` containing all unique URLs found in the input data.
            Returns an empty list if no URLs are found or if the input data
            type is not supported.
        """
        found_urls: List[str] = []

        if isinstance(data, str):
            found_urls.extend(self._extract_from_text(data))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    found_urls.extend(self._extract_from_text(item))
                else:
                    logger.warning(
                        f"URLExtractorNode received a non-string item ({type(item).__name__}) "
                        "in the input list. Skipping this item for URL extraction."
                    )
        else:
            logger.warning(
                f"URLExtractorNode received unsupported data type: {type(data).__name__}. "
                "Expected `str` or `list[str]`. Returning an empty list."
            )

        # Ensure uniqueness of URLs while preserving their original order of discovery.
        # This can be achieved by converting to a set and back to a list, or using
        # a dictionary to maintain insertion order if Python 3.7+ is guaranteed.
        unique_urls = list(dict.fromkeys(found_urls))

        if not unique_urls:
            logger.debug("No URLs were found in the provided data by URLExtractorNode.")
        else:
            logger.debug(f"Successfully extracted {len(unique_urls)} unique URL(s).")

        return unique_urls

    def _extract_from_text(self, text: str) -> List[str]:
        """
        Helper method to find all occurrences of the URL pattern in a given text string.

        Args:
            text: The string from which to extract URLs.

        Returns:
            A list of strings, where each string is a discovered URL.
        """
        return self._URL_PATTERN.findall(text)