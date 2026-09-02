import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node dedicated to extracting URLs from text data.

    This node is designed to robustly identify and extract unique HTTP and HTTPS
    URLs from either a single string or a list of strings. It employs a
    comprehensive regular expression to cover a wide range of URL formats,
    including schemes, domains, paths, queries, and fragments.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and return all unique URLs.

        The node iterates through the provided `data`, applying a regex pattern
        to find URLs. If `data` is a list, it processes each string item within
        the list. Non-string items in a list are logged as warnings and skipped.

        Args:
            data: The input data, expected to be a single string or a list of strings.
                  URLs will be extracted from the textual content.
            context: A dictionary containing contextual information for processing.
                     This node does not currently utilize the context dictionary
                     for its core logic, but it is available for future extensions.

        Returns:
            A sorted list of unique URLs found in the input data. Returns an empty
            list if no URLs are found or if the input data is empty.

        Raises:
            TypeError: If the input 'data' is not a string, a list of strings,
                       or a tuple of strings, as these are the only supported
                       input types for URL extraction.
        """
        extracted_urls_set = set()
        
        # Comprehensive regex pattern for identifying HTTP/HTTPS URLs.
        # This pattern captures common elements like scheme, domain, port, path,
        # query parameters, and fragments. It aims for broad coverage while
        # minimizing false positives.
        # Source inspiration: Modified from common URL regex patterns, e.g.,
        # from Django's URL validator or RFCs.
        url_pattern = re.compile(
            r'https?://'                                # Scheme: http:// or https://
            r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # Domain/IP and path segments
        )

        if isinstance(data, str):
            urls = url_pattern.findall(data)
            extracted_urls_set.update(urls)
            logger.debug(f"Identified {len(urls)} potential URLs from a single string input.")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    urls = url_pattern.findall(item)
                    extracted_urls_set.update(urls)
                    logger.debug(f"Identified {len(urls)} URLs from item at index {i}.")
                else:
                    logger.warning(
                        f"Skipping non-string item at index {i} in input list for URL extraction. "
                        f"Expected string, but encountered '{type(item).__name__}'."
                    )
        else:
            logger.error(
                f"Invalid input type for URLExtractorNode. Expected 'str', 'List[str]', or 'Tuple[str]', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"URLExtractorNode requires input 'data' to be a string or a sequence of strings "
                f"(list or tuple), but received '{type(data).__name__}'."
            )
        
        result = sorted(list(extracted_urls_set))
        logger.info(f"Successfully extracted {len(result)} unique URLs from the input.")
        return result