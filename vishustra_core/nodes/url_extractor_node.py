import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from text content.

    This node leverages a regular expression to identify common URL patterns
    within input strings. It can process either a single string or a list of
    strings, returning a unique list of all URLs found.
    """

    # A robust regular expression for identifying URLs.
    # It covers http/https/ftp schemes, common domain structures, and optional paths/queries/fragments.
    # This regex aims for a balance between coverage and avoiding false positives.
    _URL_PATTERN = re.compile(
        r'(?:https?|ftp):\/\/'  # Scheme (http, https, ftp)
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Subdomains and main domain
        r'[a-zA-Z]{2,6}'  # Top-level domain (e.g., com, org, net, co.uk)
        r'(?::\d{1,5})?'  # Optional port number
        r'(?:/?|[/?]\S+)'  # Optional path, query, or fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor Node"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the provided text data.

        The input `data` can be a single string or a list of strings.
        The node iterates through the text(s), applies a URL extraction regex,
        and aggregates all found URLs into a unique list.

        Args:
            data: The input data, expected to be a string or a list of strings
                  from which URLs should be extracted.
            context: A dictionary containing contextual information for processing.
                     (Not directly utilized by this specific node's logic but
                     part of the `BaseNode` interface).

        Returns:
            A list of unique URLs found in the input data. The list will be empty
            if no URLs are found or if the input data is empty.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
        """
        extracted_urls: List[str] = []

        if isinstance(data, str):
            text_inputs = [data]
        elif isinstance(data, list):
            # Ensure all items in the list are strings
            if not all(isinstance(item, str) for item in data):
                error_msg = (f"URLExtractorNode received a list containing non-string elements. "
                             f"Expected List[str], but found {type(next(item for item in data if not isinstance(item, str))).__name__}.")
                logger.error(error_msg)
                raise TypeError(error_msg)
            text_inputs = data
        else:
            error_msg = (f"URLExtractorNode received unsupported data type. "
                         f"Expected str or List[str], got {type(data).__name__}.")
            logger.error(error_msg)
            raise TypeError(error_msg)

        for text in text_inputs:
            if not text:
                logger.debug("URLExtractorNode: Skipping an empty string input.")
                continue

            found_urls = self._URL_PATTERN.findall(text)
            if found_urls:
                logger.debug(f"URLExtractorNode: Found {len(found_urls)} potential URLs in a text segment.")
                extracted_urls.extend(found_urls)

        # Use dict.fromkeys to efficiently get unique elements while preserving order
        unique_urls = list(dict.fromkeys(extracted_urls))

        logger.info(f"URLExtractorNode: Successfully extracted {len(unique_urls)} unique URLs.")
        return unique_urls
