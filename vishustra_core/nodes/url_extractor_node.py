import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text content.

    It identifies common URL patterns (http/https and www.) within the input string
    and returns them as a list of unique strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by this
                                       node but part of the signature.

        Returns:
            List[str]: A list of unique URLs found in the input data.
                       Returns an empty list if data is not a string or no URLs are found.

        Raises:
            TypeError: If the input data is not a string. (Handled gracefully with logging)
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type. Expected string, "
                f"got {type(data).__name__}. Returning an empty list of URLs."
            )
            return []

        # A robust regex pattern to capture URLs, including http(s):// and www.
        # It attempts to match common domain structures and optional paths/queries.
        url_pattern = (
            r'(?:https?://|www\.)'  # http://, https://, or www.
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain parts
            r'[a-zA-Z]{2,6}'  # TLD (e.g., .com, .org, .net)
            r'(?:/?|[/?]\S+)'  # Optional path, query, or fragment
        )

        extracted_urls: List[str] = []
        try:
            # Use re.findall to get all non-overlapping matches
            matches = re.findall(url_pattern, data)
            # Convert to a set to remove duplicates, then back to a list
            extracted_urls = list(set(matches))
            logger.debug(f"[{self.node_name}] Extracted {len(extracted_urls)} unique URLs.")
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            # On unexpected error, return an empty list to avoid propagating issues
            return []

        return extracted_urls