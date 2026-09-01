import re
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node leverages regular expressions to identify and extract all unique,
    well-formed URLs embedded within the input string. It supports common
    URL schemes like HTTP, HTTPS, and also `www.` prefixed domains.
    """

    # A comprehensive regular expression for extracting URLs.
    # This pattern aims to capture various forms of URLs, including those with
    # schemes (http/https), 'www.' prefixes, domain names, paths, and query parameters.
    _URL_REGEX = re.compile(
        r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'  # http/https with optional www
        r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'  # www without scheme
        r'https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|'  # Simpler http/https
        r'www\.[a-zA-Z0-9]+\.[^\s]{2,})' # Simpler www
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all unique URLs.

        If the input `data` is not a string, a warning is logged, and an
        empty list is returned. Otherwise, the node scans the string for URLs
        using a predefined regular expression and returns a list of unique URLs found.

        Args:
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted.
            context: A dictionary for shared state or configuration across nodes.
                     This node does not directly use the context for its core logic,
                     but it's provided as per the BaseNode interface.

        Returns:
            A list of strings, where each string is a unique URL found in the data.
            Returns an empty list if no URLs are found or if the input data
            is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Input data is not a string (type: %s). Expected a string for URL extraction. Returning an empty list.",
                self.node_name,
                type(data).__name__
            )
            return []

        # Find all occurrences of the URL pattern in the data
        extracted_urls = self._URL_REGEX.findall(data)

        # Remove duplicates by converting to a set and then back to a list
        unique_urls = list(set(extracted_urls))

        if unique_urls:
            logger.info(
                "[%s] Successfully extracted %d unique URL(s) from input data.",
                self.node_name,
                len(unique_urls)
            )
        else:
            logger.info(
                "[%s] No URLs found in the input data.",
                self.node_name
            )

        return unique_urls