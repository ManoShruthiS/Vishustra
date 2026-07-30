import re
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from unstructured text data.

    This node identifies common web URL patterns (http(s):// and www.) using a
    regular expression and then performs a light post-processing step to remove
    common trailing punctuation, aiming for clean and usable URL outputs.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract unique URLs.

        The method expects the input `data` to be a string containing the text
        from which URLs should be extracted. It uses a regular expression to find
        potential URLs and then strips common trailing punctuation that might
        have been captured due to a URL appearing at the end of a sentence.

        Args:
            data (Any): The input data. Expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing pipeline.
                                       This node does not directly utilize the context.

        Returns:
            List[str]: A list of unique, cleaned URLs as strings. Returns an empty
                       list if no URLs are found in the input data, or if the
                       input data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'. "
                "Returning an empty list of URLs."
            )
            return []

        # Regex pattern: Captures sequences of non-whitespace characters that start
        # with common web URL prefixes (http(s):// or www.).
        # This pattern is intentionally broad to ensure it catches most URLs embedded
        # in natural language text, even if they are immediately followed by punctuation.
        url_pattern = re.compile(r'(?:https?://|www\.)\S+')

        raw_extracted_urls = url_pattern.findall(data)

        if not raw_extracted_urls:
            logger.debug(f"[{self.node_name}] No URLs found in the input data.")
            return []
        
        cleaned_urls = set()
        for url in raw_extracted_urls:
            # Post-processing: URLs in natural text often have trailing punctuation
            # (e.g., "Visit example.com."). This step removes common punctuation
            # marks from the end of the extracted URL to provide a cleaner output.
            #
            # Note: This approach is pragmatic for text extraction. It might
            # inadvertently remove legitimate trailing punctuation if a URL truly
            # ends with one of these characters (e.g., 'example.com/path?param=hello!').
            # For strictly RFC-compliant URL parsing, a dedicated URL parsing library
            # (like `urllib.parse`) would be more suitable.
            cleaned_url = url.rstrip('.,!?;')
            
            # After stripping, ensure the URL still looks valid (starts with http/www.).
            # This helps filter out potential partial matches or very short invalid strings
            # that might have been produced by an over-aggressive strip, though rare.
            if cleaned_url and (cleaned_url.startswith("http") or cleaned_url.startswith("www.")):
                cleaned_urls.add(cleaned_url)

        if not cleaned_urls:
            logger.debug(f"[{self.node_name}] No unique URLs remained after cleaning.")
            return []

        logger.info(f"[{self.node_name}] Extracted {len(cleaned_urls)} unique URL(s).")
        return list(cleaned_urls)
