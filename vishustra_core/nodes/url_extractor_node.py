import re
import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from input text data.

    This node utilizes regular expressions to identify and extract common URL patterns
    (e.g., http://, https://, www.) present within a given string.
    It handles input validation and provides graceful error handling.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Args:
            data (Any): The input data, expected to be a string containing text
                        from which URLs should be extracted. If the data is not
                        a string, a warning is logged and an empty list is returned.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the node's operation. This node does not
                                       currently utilize the context directly, but it's
                                       provided for future extensibility.

        Returns:
            List[str]: A list of unique URLs found in the input data. Each URL will be
                       normalized to include a scheme (e.g., 'http://' for 'www.' URLs).
                       Returns an empty list if no URLs are found, or if the input data
                       is invalid or an error occurs during processing.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'. Returning an empty list of URLs."
            )
            return []

        # A robust regular expression to capture common URL patterns:
        # 1. URLs starting with 'http://' or 'https://'
        #    - `https?://`: Matches 'http://' or 'https://'
        #    - `[^\s<>"]+`: Matches one or more characters that are not whitespace,
        #                   angle brackets (often used in HTML tags), or double quotes.
        # 2. URLs starting with 'www.'
        #    - `www\.`: Matches 'www.'
        #    - `[^\s<>"]+`: Matches one or more non-whitespace, non-angle bracket, non-double quote characters.
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'

        try:
            # Find all non-overlapping matches of the URL pattern in the data
            found_urls = re.findall(url_pattern, data, re.IGNORECASE)

            # Normalize URLs: Prepend 'http://' to 'www.'-only URLs if a scheme is missing,
            # and then remove duplicates while preserving insertion order.
            normalized_urls = []
            seen_urls = set()
            for url in found_urls:
                if url.startswith("www.") and not url.startswith("http"):
                    # Default to http for 'www.' URLs if no scheme is specified
                    normalized_url = "http://" + url
                else:
                    normalized_url = url

                if normalized_url not in seen_urls:
                    normalized_urls.append(normalized_url)
                    seen_urls.add(normalized_url)

            if not normalized_urls:
                logger.debug(f"[{self.node_name}] No URLs found in the input data.")
            else:
                logger.debug(
                    f"[{self.node_name}] Successfully extracted {len(normalized_urls)} unique URLs."
                )

            return normalized_urls

        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            return []