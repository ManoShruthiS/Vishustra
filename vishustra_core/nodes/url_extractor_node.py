import re
import logging
from typing import Any, Dict, List

# Assume BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from text data using a robust regular expression.
    It identifies URLs starting with 'http://', 'https://', or 'www.' and attempts
    to capture their full structure including paths, query parameters, and fragments.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "url_extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input text data.

        Args:
            data (Any): The input data, expected to be a string containing text
                        from which URLs should be extracted.
            context (Dict[str, Any]): A dictionary for shared context or parameters.
                                       This node does not currently use the context.

        Returns:
            List[str]: A list of unique URLs found in the text. Returns an empty list
                       if no URLs are found or if the input data is not a string.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"URLExtractorNode expects string input, "
                f"but received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # A robust regular expression pattern for extracting URLs.
        # It covers:
        # - Schemes: http://, https://
        # - Optional 'www.' prefix for domains without a scheme
        # - Domain names (allowing hyphens and subdomains, TLDs 2-63 chars)
        # - Optional port numbers
        # - Paths, query parameters, and fragments, allowing common URL-safe characters.
        # - Word boundaries (\b) are used to prevent over-matching parts of sentences.
        # - re.IGNORECASE makes the match case-insensitive for schemes and domain components.
        url_regex = re.compile(
            r'\b(?:https?://|www\.)'                     # Scheme (http/https) or www. prefix
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+' # Domain name (e.g., example.com, sub.domain.net)
            r'(?:[a-zA-Z]{2,63})'                        # TLD (e.g., com, org, net, co.uk, gov)
            r'(?::\d{1,5})?'                             # Optional port number
            r'(?:/[-\w.~:/?#\[\]@!$&\'()*+,;=%]*)?'      # Path, query, fragment characters
            r'\b',                                       # Word boundary to end the match cleanly
            re.IGNORECASE                                # Case-insensitive matching
        )

        extracted_urls = url_regex.findall(data)
        
        # Post-process to ensure uniqueness and clean up any trailing punctuation
        # that might have been accidentally captured (e.g., URL ending a sentence).
        unique_urls: List[str] = []
        for url in extracted_urls:
            # Remove common trailing punctuation characters if they aren't part of the URL structure
            # This is a common heuristic to improve extracted URL quality from natural text.
            cleaned_url = url.rstrip('.,;!?')
            if cleaned_url not in unique_urls:
                unique_urls.append(cleaned_url)

        if not unique_urls:
            logger.info("URLExtractorNode found no URLs in the provided data.")
        else:
            logger.debug(f"URLExtractorNode successfully extracted {len(unique_urls)} unique URLs.")
        
        return unique_urls