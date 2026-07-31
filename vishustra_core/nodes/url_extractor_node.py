import re
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from a given string.
    It identifies URLs starting with 'http(s)://' or 'www.' and intelligently
    cleans potential trailing punctuation that is not part of the URL.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        This method expects `data` to be a string and will return a list
        of unique URLs found within it. Non-string inputs will result in a
        warning log and an empty list being returned.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for processing. Not directly used by this node,
                                       but available for framework consistency.

        Returns:
            List[str]: A sorted list of unique URLs found in the input data, with
                       common trailing punctuation stripped.
                       Returns an empty list if no URLs are found, or if the
                       input data is not a string or is empty.
        """
        extracted_urls: List[str] = []

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Expected a string for URL extraction. Returning an empty list."
            )
            return extracted_urls

        stripped_data = data.strip()
        if not stripped_data:
            logger.debug(f"[{self.node_name}] Received empty or whitespace-only string data. No URLs to extract.")
            return extracted_urls

        # Regex to capture potential URLs:
        # - Starts with 'http(s)://' or 'www.'
        # - Continues with any non-whitespace character (S+) until a whitespace or end of string.
        # This broad capture is intentional; trailing punctuation is handled in post-processing.
        url_pattern = re.compile(r'\b(?:https?://|www\.)\S+')

        try:
            potential_urls = url_pattern.findall(stripped_data)
            unique_urls = set()

            for url in potential_urls:
                cleaned_url = url
                
                # Strip common trailing punctuation that is often not part of a URL
                # but might be captured by the broad regex.
                # Example: "example.com." -> "example.com"
                cleaned_url = cleaned_url.rstrip('.,!?;')
                
                # Handle cases where URLs are enclosed in parentheses (e.g., in markdown links)
                # Example: "(https://example.com)" -> "https://example.com"
                if cleaned_url.startswith('(') and cleaned_url.endswith(')'):
                    cleaned_url = cleaned_url[1:-1]
                
                # Ensure the cleaned URL still has a reasonable length and structure
                # before adding it. A more advanced node might perform full URL validation.
                if cleaned_url and (cleaned_url.startswith('http') or cleaned_url.startswith('www.')):
                    unique_urls.add(cleaned_url)

            extracted_urls = sorted(list(unique_urls))

            if extracted_urls:
                logger.debug(f"[{self.node_name}] Successfully extracted {len(extracted_urls)} unique URLs.")
            else:
                logger.debug(f"[{self.node_name}] No URLs found in the provided data.")

        except re.error as e:
            logger.error(f"[{self.node_name}] Regular expression error during URL extraction: {e}")
        except Exception as e:
            logger.error(f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}", exc_info=True)

        return extracted_urls