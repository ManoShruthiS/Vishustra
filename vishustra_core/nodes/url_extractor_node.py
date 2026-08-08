import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node is available in the Python path
# For local development/testing, you might need to adjust this import,
# but for the framework context, this is the correct relative path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node uses regular expressions to find common URL patterns
    (e.g., http(s)://, www.) within the input string data.
    """

    # A robust regex for common URL patterns, including http(s):// and www.
    # It accounts for various characters allowed in URLs, including path, query,
    # and fragment components.
    _URL_REGEX = re.compile(
        r'\b(?:https?://|www\.)'  # Scheme or www. prefix
        r'(?:[a-zA-Z0-9][-a-zA-Z0-9]*\.)+'  # Domain name (e.g., example.com)
        r'[a-zA-Z]{2,63}'  # Top-level domain (e.g., com, org)
        r'(?::\d{1,5})?'  # Optional port number
        r'(?:[/](?:[a-zA-Z0-9\-._~!$&\'()*+,;=:@/?#%]*))*\b'  # Path, query, fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data: The input data, expected to be a string or something
                  that can be converted to a string.
            context: A dictionary containing contextual information for the process.

        Returns:
            A list of unique URLs found in the data. Returns an empty list
            if no URLs are found or if the input data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Expected a string for URL extraction. Returning an empty list."
            )
            return []

        try:
            # Find all unique URLs matching the regex pattern
            extracted_urls = list(set(self._URL_REGEX.findall(data)))
            
            if not extracted_urls:
                logger.debug(f"[{self.node_name}] No URLs found in the provided data.")
            else:
                logger.info(f"[{self.node_name}] Successfully extracted {len(extracted_urls)} unique URLs.")
            
            return extracted_urls
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            # Depending on desired error handling, might re-raise or return empty list.
            # Returning empty list is generally safer in a pipeline for an extractor.
            return []

