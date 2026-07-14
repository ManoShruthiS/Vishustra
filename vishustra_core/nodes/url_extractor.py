import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A processing node that extracts URLs from input text data.

    It identifies URLs starting with 'http://', 'https://', or 'www.'
    and returns them as a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing context information
                                       for the current processing flow.

        Returns:
            List[str]: A list of extracted URLs. Returns an empty list if no URLs
                       are found or if the input data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Invalid input data type. Expected 'str', but received '%s'. "
                "Returning an empty list.",
                self.node_name, type(data).__name__
            )
            return []

        # Regular expression to match URLs starting with http(s):// or www.
        # This pattern aims to be robust but avoids being overly greedy
        # by not matching trailing punctuation unless it's part of the URL.
        # It captures common URL structures and prevents matching parts of words.
        url_pattern = re.compile(
            r'\b(?:https?://|www\.)'  # Start with http://, https://, or www.
            r'(?:[a-zA-Z0-9-._~:/?#[\]@!$&\'()*+,;=]|%[0-9a-fA-F]{2})+'  # URL characters
            r'(?<![.,;])' # Negative lookbehind to not include trailing punctuation (like at end of sentence)
        )
        
        extracted_urls = url_pattern.findall(data)

        if extracted_urls:
            logger.debug(
                "[%s] Successfully extracted %d URL(s). First URL: %s",
                self.node_name, len(extracted_urls), extracted_urls[0]
            )
        else:
            logger.debug("[%s] No URLs found in the input data.", self.node_name)
            
        return extracted_urls

