import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# Regex pattern to identify URLs.
# This pattern aims to capture http(s):// and www. prefixed URLs,
# followed by a domain name and an optional path/query/fragment.
# It's designed to be reasonably robust for common URL patterns without being overly complex.
URL_REGEX = re.compile(
    r'(?:https?://|www\.)'  # Match 'http://', 'https://', or 'www.'
    r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Match domain components (e.g., example.com, sub.domain.co.uk)
    r'[a-zA-Z]{2,6}'  # Match TLDs (e.g., com, org, co.uk)
    r'(?:/[^"\s]*)?'  # Optionally match path, query, or fragment, excluding quotes and spaces.
)


class UrlExtractorNode(BaseNode):
    """
    A processing node that extracts all URLs from an input string.

    It identifies URLs conforming to common patterns (http(s):// or www.)
    and returns them as a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "UrlExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data (Any): The input data, expected to be a string containing text.
                        If not a string, an empty list will be returned after logging a warning.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. Not directly used
                                       by this node, but passed along.

        Returns:
            List[str]: A list of unique URLs found in the input data.
                       Returns an empty list if no URLs are found or if the input
                       data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type for URL extraction. "
                f"Expected string, got {type(data).__name__}. Returning empty list."
            )
            return []

        logger.debug(f"[{self.node_name}] Starting URL extraction from input data.")

        try:
            # Find all unique URLs matching the pattern
            found_urls = list(set(URL_REGEX.findall(data)))
            logger.info(f"[{self.node_name}] Found {len(found_urls)} unique URLs.")
            return found_urls
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            return []