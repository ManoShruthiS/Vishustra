
import re
import logging
from typing import Any, Dict, List, Union, Iterable, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node utilizes a robust regular expression to find and extract URLs
    (supporting http, https, and www prefixes) from a given string or an
    iterable of strings. It ensures that only unique URLs are returned.
    """

    # Comprehensive regular expression for identifying URLs.
    # This regex is designed to capture URLs starting with http(s):// or www.
    # It covers common URL structures, including various allowed characters in
    # paths, query parameters, and anchors, while aiming to minimize false positives.
    # The pattern is based on widely adopted robust URL matching techniques.
    _URL_REGEX = re.compile(
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, Iterable[str], Any], context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract unique URLs.

        The `data` input can be a single string or an iterable of strings.
        Non-string elements within an iterable will be skipped with a debug log.
        If the primary `data` input itself is not a string or an iterable, a warning
        is logged, and an empty list is returned.

        Args:
            data: The input data, expected to be a string or an iterable of strings,
                  from which URLs are to be extracted.
            context: A dictionary containing contextual information for the processing
                     pipeline (not directly utilized by this node, but passed along).

        Returns:
            A list of unique URLs found in the input data, sorted alphabetically for
            deterministic output. Returns an empty list if no URLs are found or if
            the input data is invalid.
        """
        extracted_urls: Set[str] = set()

        if data is None:
            logger.debug("Received None data for URL extraction. Returning an empty list.")
            return []

        if isinstance(data, str):
            urls_found = self._URL_REGEX.findall(data)
            # The regex returns tuples because of multiple capturing groups.
            # The first element of each tuple is the full match.
            extracted_urls.update(url_match[0] for url_match in urls_found)
        elif isinstance(data, Iterable):
            for item in data:
                if isinstance(item, str):
                    urls_found = self._URL_REGEX.findall(item)
                    extracted_urls.update(url_match[0] for url_match in urls_found)
                else:
                    logger.debug(
                        f"Skipping non-string item of type '{type(item).__name__}' "
                        "within iterable data during URL extraction."
                    )
        else:
            logger.warning(
                f"Unsupported data type '{type(data).__name__}' for URL extraction. "
                "Expected 'str' or 'Iterable[str]'. Returning an empty list."
            )
            return []

        # Convert the set of unique URLs to a sorted list for consistent output.
        return sorted(list(extracted_urls))

