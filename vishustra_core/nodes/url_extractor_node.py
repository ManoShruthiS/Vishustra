import logging
import re
from typing import Any, Dict, List, Union, Set

# Vishustra core base node for inheritance
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts unique URLs from text content.

    This node identifies URLs conforming to standard HTTP/HTTPS schemes.
    It can process a single string or a list of strings, returning a list of
    all unique URLs found across the input.
    """

    # Compiled regex pattern for robust URL extraction.
    # This pattern is designed to capture common HTTP/HTTPS URLs including
    # hostnames, IP addresses, ports, paths, query parameters, and fragments.
    # It accounts for various URL-safe characters specified in RFC 3986.
    _URL_REGEX = re.compile(
        r'http[s]?://'  # Match http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # Match one or more URL-safe characters including domain, path, query, fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        The input `data` can be a single string or a list of strings.
        - If `data` is a string, URLs are extracted directly from it.
        - If `data` is a list of strings, URLs are extracted from each string
          in the list. Non-string elements within the list are logged as
          warnings and skipped to prevent processing errors.

        All extracted URLs are collected, deduplicated, and returned in a
        sorted list.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     (This node does not utilize the context but adheres to the
                     BaseNode interface by accepting it).

        Returns:
            A list of unique URLs found in the input data, sorted alphabetically.

        Raises:
            TypeError: If the input `data` is neither a string nor a list of strings,
                       indicating an unsupported input type for this node.
        """
        extracted_urls: Set[str] = set()

        if isinstance(data, str):
            logger.debug("URLExtractorNode received single string data for processing.")
            found_urls = self._URL_REGEX.findall(data)
            extracted_urls.update(found_urls)
        elif isinstance(data, list):
            logger.debug("URLExtractorNode received a list of items for processing.")
            for i, item in enumerate(data):
                if isinstance(item, str):
                    found_urls = self._URL_REGEX.findall(item)
                    extracted_urls.update(found_urls)
                else:
                    logger.warning(
                        f"URLExtractorNode: Skipping non-string item at index {i} in input list. "
                        f"Expected string, but received type '{type(item).__name__}'."
                    )
        else:
            logger.error(
                f"URLExtractorNode received invalid input data type. "
                f"Expected 'str' or 'List[str]', but got '{type(data).__name__}'."
            )
            raise TypeError(
                f"URLExtractorNode expects 'data' to be a string or a list of strings, "
                f"but received type '{type(data).__name__}'."
            )

        result_list = sorted(list(extracted_urls))
        logger.info(f"URLExtractorNode successfully extracted {len(result_list)} unique URLs.")
        return result_list