import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from input text.
    It identifies common HTTP/HTTPS URLs within the provided string data,
    including those with domain names or IPv4 addresses, optional 'www.' prefix,
    ports, paths, query parameters, and fragments.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This can
                                       include session details or global settings.

        Returns:
            List[str]: A list of full URLs found in the input data. Returns an empty
                       list if no URLs are found or if the input data is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting URL extraction. Context keys: {list(context.keys())}")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is not a string. Received type: {type(data).__name__}. "
                "Returning an empty list as no URLs can be extracted from non-string data."
            )
            return []

        # Regex components for robust URL matching:
        # 1. IPv4 address pattern
        ipv4_pattern = r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'

        # 2. Domain name pattern (including subdomains and flexible TLDs)
        # TLDs are typically 2+ characters, allowing up to 10 for future proofing and new TLDs.
        domain_pattern = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+' \
                         r'[a-zA-Z]{2,10}'

        # 3. Combine hostname (domain or IPv4)
        host_pattern = f'(?:{domain_pattern}|{ipv4_pattern})'

        # 4. Full URL regex combining all parts
        # - https?://          : HTTP or HTTPS scheme (required for strict URL extraction)
        # - (?:www\.)?         : Optional 'www.' prefix
        # - {host_pattern}     : The combined domain name or IPv4
        # - (?::\d{1,5})?      : Optional port number (1 to 5 digits)
        # - (?:/?|[/?][^\s]*)  : Optional path, query, and fragment (allowing non-whitespace characters)
        url_regex = re.compile(
            r'https?://'
            r'(?:www\.)?'
            f'{host_pattern}'
            r'(?::\d{1,5})?'
            r'(?:/?|[/?][^\s]*)'
            , re.IGNORECASE
        )

        found_urls: List[str] = url_regex.findall(data)

        if found_urls:
            logger.debug(f"[{self.node_name}] Successfully extracted {len(found_urls)} URLs.")
        else:
            logger.debug(f"[{self.node_name}] No URLs found in the input data.")

        return found_urls