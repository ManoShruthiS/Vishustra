
import logging
import re
from typing import Any, Dict, List

# Assuming vishustra_core is a package where BaseNode is defined
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text content.

    It uses a regular expression to identify and return a list of URLs
    found within the input data.
    """

    # A robust regular expression pattern for matching URLs.
    # This pattern attempts to capture various valid URL formats, including
    # those with and without protocols (http/https) and subdomains.
    # It also handles different TLDs and common path characters.
    _URL_PATTERN = re.compile(
        r'(?:https?://|www\.)'  # Protocol (http/https:// or www.)
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+' # Domain name part
        r'(?:[a-zA-Z]{2,6}\.?)' # TLD (e.g., com, org, net, co.uk)
        r'(?:/?|[/?]\S+)',      # Optional path (including query params, fragments)
        re.IGNORECASE           # Case-insensitive matching for domain/protocol
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Args:
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted. If not a string,
                  a warning is logged, and an empty list is returned.
            context: A dictionary containing contextual information
                     for the processing flow. Not directly used by this node,
                     but passed along.

        Returns:
            A list of unique URLs found in the input data. Returns an empty
            list if no URLs are found or if the input data is invalid.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'. Returning empty list."
            )
            return []

        try:
            # Find all matches of the URL pattern in the input string
            found_urls = self._URL_PATTERN.findall(data)

            # Post-process: Add 'https://' to URLs starting with 'www.' if missing
            # and ensure uniqueness.
            processed_urls = []
            for url in found_urls:
                if url.startswith('www.') and not url.startswith(('http://', 'https://')):
                    processed_urls.append(f"https://{url}")
                else:
                    processed_urls.append(url)
            
            # Use a set to maintain uniqueness and then convert back to a list
            unique_urls = sorted(list(set(processed_urls)))

            if unique_urls:
                logger.debug(f"[{self.node_name}] Successfully extracted {len(unique_urls)} unique URLs.")
            else:
                logger.info(f"[{self.node_name}] No URLs found in the provided data.")

            return unique_urls
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            return []

