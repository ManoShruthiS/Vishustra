import re
import logging
from typing import Any, Dict, List

# Importing the base node from the Vishustra core nodes module.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract HTTP/HTTPS URLs from textual data.

    This node utilizes regular expressions to identify potential URLs and
    includes post-processing to strip common trailing punctuation, ensuring
    clean and accurate URL extraction.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts HTTP/HTTPS URLs from the provided input data.

        The `data` input is expected to be a string. If the input is not a string,
        a warning will be logged, and an empty list will be returned to ensure
        pipeline stability. The `context` dictionary is included as per the
        `BaseNode` interface but is not directly used by this specific node.

        Args:
            data (Any): The input data, which should be a string containing text
                        from which URLs are to be extracted.
            context (Dict[str, Any]): A dictionary holding contextual information
                                       relevant to the current processing flow.

        Returns:
            List[str]: A sorted list of unique HTTP/HTTPS URLs found in the
                       input string. Returns an empty list if no URLs are found
                       or if the input data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Expected string for URL extraction. Returning an empty list."
            )
            return []

        # Find all occurrences of HTTP/HTTPS URLs.
        # This regex broadly captures strings starting with 'http://' or 'https://'
        # followed by one or more non-whitespace characters.
        raw_urls = re.findall(r'https?://\S+', data, re.IGNORECASE)

        cleaned_urls = []
        for url in raw_urls:
            # Define common trailing punctuation characters that are often
            # mistakenly included when extracting URLs from text.
            trailing_punctuation = '.,;!?"\'<>(){}[]'
            
            # Iteratively remove trailing punctuation until the URL ends
            # with a character not in the punctuation set, or the URL becomes empty.
            while url and url[-1] in trailing_punctuation:
                url = url[:-1]
            
            # After cleaning, ensure the remaining string still starts like a URL
            # and is not empty before adding it to the list.
            if url and re.match(r'https?://', url, re.IGNORECASE):
                cleaned_urls.append(url)
        
        # Convert the list to a set to eliminate duplicate URLs, then convert back
        # to a sorted list for consistent output order.
        unique_and_sorted_urls = sorted(list(set(cleaned_urls)))
        
        if unique_and_sorted_urls:
            logger.debug(f"[{self.node_name}] Successfully extracted {len(unique_and_sorted_urls)} unique URLs.")
        else:
            logger.debug(f"[{self.node_name}] No URLs were found in the input data after processing.")

        return unique_and_sorted_urls
