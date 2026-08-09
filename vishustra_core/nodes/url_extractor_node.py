import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from text data.

    This node identifies common URL patterns within the input string, including
    those starting with 'http://', 'https://', or 'www.'.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all identifiable URLs.

        Args:
            data: The input data, expected to be a string containing text.
                  If `data` is not a string, a warning is logged, and an empty list is returned.
            context: A dictionary of contextual information, which is not directly
                     used by this node but is part of the `BaseNode` interface.

        Returns:
            A list of strings, where each string is a URL found in the input data.
            Returns an empty list if no URLs are found or if the input data is
            not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data. Expected a string for URL extraction. "
                "Type received: %s. Returning an empty list.",
                type(data)
            )
            return []

        # Regular expression to broadly match URLs starting with http(s):// or www.
        # and continuing with non-whitespace characters. This pattern is a good
        # balance for general URL extraction without being overly complex for
        # initial implementation.
        url_pattern = r'https?://\S+|www\.\S+'

        try:
            extracted_urls = re.findall(url_pattern, data)
            if extracted_urls:
                logger.debug("Successfully extracted %d URLs from data.", len(extracted_urls))
            else:
                logger.debug("No URLs found in the provided data.")
            return extracted_urls
        except Exception as e:
            logger.error(
                "An unexpected error occurred during URL extraction in URLExtractorNode: %s",
                e,
                exc_info=True
            )
            return []