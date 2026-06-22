import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from text data using a robust regular expression.

    This node identifies URLs that start with 'http://', 'https://', or 'www.',
    including various domain structures, paths, and query parameters, while
    attempting to correctly handle common trailing punctuation.
    """

    # A comprehensive regex pattern to match URLs.
    # It covers:
    # - Optional scheme (http:// or https://)
    # - Optional 'www.' subdomain
    # - Domain name with alphanumeric characters and hyphens, followed by a TLD (2-6 chars)
    # - Optional path, query, and fragment parts, handling various special characters
    #   and attempting to exclude common trailing punctuation marks like '.', ',', ')'
    _URL_REGEX = re.compile(
        r'\b(?:https?://|www\.)'  # Start with http(s):// or www.
        r'(?:[a-zA-Z0-9-]+\.)+'   # One or more domain parts (e.g., example.)
        r'[a-zA-Z]{2,6}'          # Top-level domain (e.g., com, org, net)
        r'(?:'                    # Start of optional path/query/fragment group
        r'/?'                     # Optional leading slash for path
        r'[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]*' # Path characters, excluding common delimiters
        r'[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]' # Ensure it doesn't end with a delimiter
        r')?'                     # End of optional path/query/fragment group
        r'\b'                     # Word boundary to prevent partial matches
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        The node expects the input `data` to be a string. If `data` is not a string,
        a warning is logged, and an empty list is returned.

        Args:
            data (Any): The input data, expected to be a string containing text
                        from which URLs should be extracted.
            context (Dict[str, Any]): A dictionary for shared context or configuration.
                                     (Not directly used by this node but part of the signature).

        Returns:
            List[str]: A sorted list of unique URLs found in the data.
                       Returns an empty list if no URLs are found or if the input
                       data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data type ({type(data).__name__}). "
                "Expected string for URL extraction. Returning empty list."
            )
            return []

        extracted_urls: Set[str] = set()
        try:
            # Find all matches of the URL pattern in the data
            for match in self._URL_REGEX.finditer(data):
                url = match.group(0)
                extracted_urls.add(url)

            logger.info(f"[{self.node_name}] Successfully extracted {len(extracted_urls)} unique URLs.")
            # Convert set to list and sort for consistent output
            return sorted(list(extracted_urls))

        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            return []