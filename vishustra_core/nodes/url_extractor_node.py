import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from input text.

    This node expects the input 'data' to be a string and uses a robust regular expression
    to find and return a list of all URLs (starting with http(s):// or www.) present in the text.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data: The input data, expected to be a string containing text from which URLs should be extracted.
            context: A dictionary of contextual information. This node does not currently utilize the context.

        Returns:
            A list of strings, where each string is a URL found in the input data.
            Returns an empty list if no URLs are found or if the input data is an empty string.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Input data must be a string for URL extraction. "
                f"Received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data:
            logger.debug(f"[{self.node_name}] Received empty string data. Returning an empty list of URLs.")
            return []

        # A robust regular expression to capture URLs.
        # This regex broadly covers common URL patterns including scheme (http/https),
        # domain names (with subdomains, TLDs), optional port, path, query parameters,
        # and fragments. It also tries to avoid capturing trailing punctuation often
        # found immediately after a URL in natural language text.
        url_regex = re.compile(
            r'\b(?:https?://|www\.)'  # Scheme (http/https) or www.
            r'(?:[a-zA-Z0-9-]+\.)+'   # Domain name parts (e.g., example.com, sub.domain.co.uk)
            r'[a-zA-Z]{2,63}'         # Top-level domain (e.g., com, org, net, info, co.uk, gov)
            r'(?::\d{1,5})?'          # Optional port number (e.g., :8080)
            r'(?:/?|[/?]\S+)'         # Path, query string, fragment (e.g., /path?q=test#frag)
            r'(?<![.,;])\b'           # Negative lookbehind to not capture trailing punctuation,
                                      # followed by a word boundary.
        )

        extracted_urls = url_regex.findall(data)

        if extracted_urls:
            logger.info(f"[{self.node_name}] Successfully extracted {len(extracted_urls)} URLs.")
            logger.debug(f"[{self.node_name}] Extracted URLs: {extracted_urls}")
        else:
            logger.info(f"[{self.node_name}] No URLs were found in the provided data.")

        return extracted_urls