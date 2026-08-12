import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.
    It identifies common HTTP/HTTPS URLs and returns them as a list of strings.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to extract URLs.

        This method expects a string as input data and uses a regular expression
        to find common HTTP/HTTPS URLs within the text.

        Args:
            data: The input data, expected to be a string containing text.
                  If a non-string type is provided, a warning is logged, and
                  an empty list is returned.
            context: A dictionary of contextual information. This node does not
                     currently utilize any context.

        Returns:
            A list of unique URLs (str) found in the input data.
            Returns an empty list if data is not a string or no URLs are found.
            The return type matches `Any` as per the `BaseNode` definition.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Expected string for URL extraction. Returning empty list."
            )
            return []

        # A robust regular expression to find common URLs in text.
        # This pattern aims to capture full URLs including scheme, domain,
        # and optional path/query/fragment components.
        # Explanation of the regex:
        # (?:(?:https?|ftp):\/\/|www\.) - Matches 'http://', 'https://', 'ftp://', or 'www.'
        # (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+ - Matches domain name segments (e.g., 'sub.domain.')
        # [a-zA-Z]{2,6} - Matches the top-level domain (e.g., 'com', 'org', 'co.uk')
        # (?:[-\w._~:/?#\[\]@!$&'()*+,;=%]*)? - Matches optional path, query, fragment, etc.
        url_regex = re.compile(
            r'(?:(?:https?|ftp):\/\/|www\.)'  # Scheme (http/https/ftp) or 'www.' prefix
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain name parts (e.g., 'example.com')
            r'[a-zA-Z]{2,6}'  # Top-level domain (e.g., 'com', 'org', 'net')
            r'(?:[-\w._~:/?#\[\]@!$&\'()*+,;=%]*)?'  # Optional path, query, fragment, and other URL characters
        )

        found_urls = url_regex.findall(data)

        # Remove duplicate URLs using a set, then convert back to a list.
        unique_urls: List[str] = list(set(found_urls))

        if not unique_urls:
            logger.debug(f"[{self.node_name}] No URLs found in the provided text.")
        else:
            logger.info(f"[{self.node_name}] Extracted {len(unique_urls)} unique URL(s).")

        return unique_urls