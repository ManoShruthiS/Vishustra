import re
import logging
from typing import Any, Dict, List, Set, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from textual data.

    This node leverages regular expressions to identify and collect common URL patterns,
    including those with HTTP/HTTPS schemes and 'www.' prefixes, from a variety of
    input data structures.
    """

    # A compiled regular expression for robust URL matching.
    # This pattern aims to capture URLs beginning with 'https://', 'http://', or 'www.',
    # followed by domain names, paths, queries, and fragments, while being mindful
    # of common URL character sets. It is case-insensitive.
    _URL_REGEX = re.compile(
        r'(?:https?://|www\.)'  # Scheme (http/https) or 'www.' prefix
        r'[-a-zA-Z0-9@:%._\+~#=]{1,256}'  # Domain name and optional subdomains
        r'\.[a-zA-Z0-9()]{1,6}\b'  # Top-level domain (e.g., .com, .org, .co.uk)
        r'(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',  # Optional path, query, fragment
        re.IGNORECASE
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all unique URLs.

        The input `data` can be:
        - A single string: URLs are extracted directly from this string.
        - An iterable (e.g., list, tuple, set) of strings: Each string element
          within the iterable is processed for URL extraction. Non-string elements
          in the iterable will be noted with a warning and skipped.
        - `None`: Returns an empty list with a debug log.
        - Any other type: Logs a warning and returns an empty list, as it's not
          a supported input type for URL extraction.

        Args:
            data (Any): The input data expected to contain text from which URLs
                        should be extracted.
            context (Dict[str, Any]): A dictionary providing contextual information
                                      for the current processing pipeline.
                                      This node does not directly utilize the context
                                      but adheres to the `BaseNode` interface.

        Returns:
            List[str]: A sorted list of unique URLs found in the input data.
                       Returns an empty list if no URLs are found or if the
                       input data is not in a processable format.
        """
        extracted_urls: Set[str] = set()

        if data is None:
            logger.debug("Received None data for URL extraction. Returning an empty list.")
            return []

        if isinstance(data, str):
            urls = self._URL_REGEX.findall(data)
            extracted_urls.update(urls)
        elif isinstance(data, (list, tuple, set)):
            for index, item in enumerate(data):
                if isinstance(item, str):
                    urls = self._URL_REGEX.findall(item)
                    extracted_urls.update(urls)
                else:
                    logger.warning(
                        "Element at index %d (type '%s') in iterable data is not a string. Ignoring it for URL extraction.",
                        index, type(item).__name__
                    )
        else:
            logger.warning(
                "Input data type '%s' is not supported for URL extraction. "
                "Expected a string or an iterable of strings. Returning an empty list.",
                type(data).__name__
            )
            return []

        if not extracted_urls:
            logger.debug("No URLs were found in the provided data after processing.")

        # Convert the set of unique URLs to a sorted list for consistent and predictable output.
        return sorted(list(extracted_urls))