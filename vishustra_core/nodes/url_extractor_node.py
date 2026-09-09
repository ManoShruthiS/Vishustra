import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from text content.

    This node leverages regular expressions to identify common URL patterns
    (e.g., those starting with `http(s)://` or `www.`) within the provided
    input data. It's robust against various text formats and ensures
    that only unique URLs are returned.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL_Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract URLs.

        Expects the input `data` to be a string. If a non-string type is
        encountered, a warning is logged, and an empty list is returned,
        preventing potential downstream errors.

        Args:
            data: The input content, typically a string from which URLs
                  are to be extracted.
            context: A dictionary holding contextual information relevant
                     to the current processing pipeline. This node does
                     not directly use the context but adheres to the
                     BaseNode interface.

        Returns:
            A list of unique URLs found within the `data`. The order of
            URLs in the list corresponds to their first appearance in the
            input text. Returns an empty list if no URLs are found or
            if the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data "
                f"(type: {type(data).__name__}). Expected a string for URL "
                "extraction. Returning an empty list."
            )
            return []

        # A robust regex pattern to capture URLs starting with 'http://',
        # 'https://', or 'www.'. It broadly matches characters typically
        # found in URLs and avoids breaking on common delimiters like
        # whitespace, angle brackets, or double quotes.
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        
        extracted_urls = re.findall(url_pattern, data)
        
        # Convert to an ordered set (using dict.fromkeys) to ensure
        # uniqueness while preserving the original order of appearance.
        unique_urls = list(dict.fromkeys(extracted_urls))

        if unique_urls:
            logger.debug(
                f"[{self.node_name}] Successfully extracted "
                f"{len(unique_urls)} unique URLs from the input data."
            )
        else:
            logger.debug(f"[{self.node_name}] No URLs found in the provided data.")

        return unique_urls