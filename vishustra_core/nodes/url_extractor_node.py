import logging
import re
from typing import Any, Dict, List, Set

# BaseNode is expected to be in this path based on the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A processing node designed to extract URLs from textual content.

    This node expects a string as input. It utilizes a regular expression
    to identify and return a list of unique URLs. The extraction mechanism
    is robust enough to capture common URL formats, including those prefixed
    with 'http://', 'https://', and 'www.', and properly handles various
    URL-valid characters in the path, query, and fragment.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data: The input data, expected to be a string.
            context: A dictionary containing contextual information
                     (not directly used by this node but part of the BaseNode interface).

        Returns:
            A list of unique URLs found in the input text. Returns an empty list
            if no URLs are found.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"URLExtractorNode received invalid input type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # A robust regex pattern to find URLs.
        # It matches 'http://', 'https://', or 'www.'
        # followed by a sequence of characters commonly found in URLs,
        # including alphanumeric, various punctuation, and URL-encoded characters.
        # The use of word boundaries `\b` helps prevent partial matches and
        # handles common trailing punctuation from sentences.
        url_pattern = re.compile(
            r'\b'  # Word boundary at the start
            r'(?:https?://|www\.)'  # Match 'http://', 'https://', or 'www.' prefix
            r'(?:'                  # Start non-capturing group for URL body characters
            r'[a-zA-Z0-9$-_@.&+]|'  # Common URL-safe characters
            r'[!*\(\),]|'           # Additional punctuation allowed in URLs
            r'%[0-9a-fA-F]{2}'      # URL-encoded characters (e.g., %20 for space)
            r')+'                   # Match one or more of the above URL body characters
            r'\b'  # Word boundary at the end
        )

        extracted_urls: Set[str] = set()
        try:
            found_urls = url_pattern.findall(data)
            for url in found_urls:
                extracted_urls.add(url)
        except Exception as e:
            logger.error(f"An unexpected error occurred during URL extraction: {e}", exc_info=True)
            # In case of a regex engine error, we return an empty list gracefully
            return []

        if extracted_urls:
            logger.info(f"Successfully extracted {len(extracted_urls)} unique URLs.")
        else:
            logger.info("No URLs found in the provided text data.")

        # Convert the set of unique URLs to a sorted list for consistent, deterministic output.
        return sorted(list(extracted_urls))