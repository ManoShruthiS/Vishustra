import logging
import re
from typing import Any, Dict, List

# Assuming this import path from the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node identifies and extracts valid URLs (http, https, and www-prefixed)
    from an input string. It uses a robust regular expression to cover common
    URL formats including protocols, domain names, paths, and query parameters.
    If the input data is not a string, it logs a warning and returns an empty list
    to maintain a consistent output type.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        The method expects the input `data` to be a string. It will scan this
        string for patterns matching typical URL structures.

        Args:
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted.
            context: A dictionary containing contextual information for the node's
                     operation. This node does not explicitly use the context
                     but it's part of the BaseNode interface.

        Returns:
            A list of strings, where each string is a detected URL.
            Returns an empty list if no URLs are found in the input string,
            or if the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data for URL extraction. "
                f"Type: {type(data).__name__}. Returning empty list."
            )
            return []

        # A comprehensive but balanced regex for common URL formats:
        # - Starts with http://, https://, or www.
        # - Followed by domain components (alphanumeric, hyphens, dots).
        # - Ends with a TLD of 2-6 characters.
        # - Optionally includes path, query parameters, and fragments (non-whitespace characters).
        # The \b ensures word boundaries to avoid partial matches within other words.
        url_pattern = re.compile(
            r'\b(?:https?://|ftp://|www\.)'  # Protocols or www.
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain parts (sub.domain.com)
            r'[a-zA-Z]{2,6}'  # Top-Level Domain (e.g., com, org, net, co.uk)
            r'(?:/?|[/?]\S+)'  # Optional path, query, fragment
            r'\b'
        )

        extracted_urls = url_pattern.findall(data)
        logger.debug(f"[{self.node_name}] Extracted {len(extracted_urls)} URLs from input data.")
        return extracted_urls