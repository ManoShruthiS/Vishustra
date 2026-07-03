import re
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A processing node designed to extract URLs from textual content.

    This node expects a string as its input `data` and employs a robust
    regular expression to identify and return a list of all detected URLs.
    It supports common URL schemes including HTTP, HTTPS, FTP, and URLs
    prefixed with 'www.'.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts all identifiable URLs from the input data.

        Args:
            data: The input data, ideally a string containing the text
                  from which URLs are to be extracted.
            context: A dictionary containing contextual information relevant
                     to the current processing flow. Not directly used by
                     this node, but required by the BaseNode interface.

        Returns:
            A list of strings, where each string is a fully qualified URL
            found within the input `data`. Returns an empty list if no URLs
            are found, or if the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data (type: %s). "
                "Expected a string for URL extraction. Returning an empty list.",
                type(data).__name__
            )
            return []

        # A comprehensive regular expression pattern to match URLs.
        # This pattern captures URLs starting with common schemes (http, https, ftp)
        # or 'www.', followed by a sequence of characters typically allowed
        # in URLs, including various symbols for paths, queries, fragments,
        # and authentication components.
        url_pattern = re.compile(
            r'(?:(?:https?|ftp):\/\/|www\.)'  # Scheme (http, https, ftp) or www.
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+' # Domain name (includes subdomains)
            r'[a-zA-Z]{2,6}' # TLD (e.g., com, org, net, co.uk)
            r'(?:[^\s<>"]*)?' # Optional path, query, fragment (any char except whitespace, <, > or ")
            r'(?=\s|$|[,.;])' # Positive lookahead for whitespace, end of string, or common punctuation
        )

        try:
            urls = url_pattern.findall(data)
            if urls:
                logger.debug("Successfully extracted %d URLs.", len(urls))
            else:
                logger.debug("No URLs found in the provided data.")
            return urls
        except Exception as e:
            logger.error(
                "An unexpected error occurred during URL extraction in URLExtractorNode: %s",
                e, exc_info=True
            )
            return []