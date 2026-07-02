import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from textual data.
    It supports processing single strings or collections of strings.
    URLs are identified using a common regular expression pattern.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract URLs.

        The method expects the input `data` to be either a single string
        or an iterable (like a list or tuple) containing strings.
        Non-string elements within an iterable will be skipped with a warning.
        Any other unsupported data types will result in a warning and
        an empty list being returned.

        Args:
            data: The input data to be scanned for URLs.
            context: A dictionary containing contextual information
                     (this node does not utilize the context).

        Returns:
            A list of unique URLs found in the input data.
        """
        extracted_urls: List[str] = []
        # A robust regex pattern for matching http(s):// or www. URLs.
        # It aims to cover common URL structures while avoiding over-matching.
        url_pattern = re.compile(
            r'(https?://[^\s<>"]+|www\.[^\s<>"]+)'
        )

        if isinstance(data, str):
            urls = url_pattern.findall(data)
            extracted_urls.extend(urls)
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    urls = url_pattern.findall(item)
                    extracted_urls.extend(urls)
                else:
                    logger.warning(
                        f"URLExtractorNode: Encountered non-string item at index {i} "
                        f"in input list. Expected string for URL extraction, "
                        f"but received {type(item).__name__}. This item will be skipped."
                    )
        else:
            logger.warning(
                f"URLExtractorNode: Received an unsupported data type for processing. "
                f"Expected 'str' or an iterable of 'str', but got {type(data).__name__}. "
                f"Returning an empty list of URLs."
            )

        # Return a list of unique URLs found to avoid duplicates
        return list(set(extracted_urls))