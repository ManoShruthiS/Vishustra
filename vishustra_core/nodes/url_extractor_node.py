import logging
import re
from typing import Any, Dict, List

# Importing BaseNode from the project's core module structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from text data.
    It identifies common URL patterns including HTTP(S) and www-prefixed links,
    and returns a list of unique URLs found.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all found URLs.
        
        Args:
            data (Any): The input data, expected to be a string containing text
                        from which URLs should be extracted.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. This node does not currently
                                       utilize the context, but it adheres to the
                                       `BaseNode` interface.

        Returns:
            List[str]: A list of unique URLs found in the input text.
                       Returns an empty list if no URLs are found, or if the input
                       data is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting URL extraction process.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Expected a string for URL extraction. Returning an empty list."
            )
            return []

        # A robust regular expression to capture various forms of URLs:
        # - Matches URLs starting with http://, https://, or www.
        # - Handles domain names, TLDs, paths, query parameters, and fragments.
        # - Accounts for potential punctuation directly following a URL in text.
        url_pattern = re.compile(
            r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        )
        
        found_matches = url_pattern.findall(data)
        
        # The regex can produce tuples if there are multiple capturing groups.
        # The first element of each tuple is typically the full match.
        # We also ensure uniqueness and strip common trailing punctuation.
        extracted_urls_set = set()
        for match_group in found_matches:
            # The first element of the tuple corresponds to the full URL matched by the outer group
            url = match_group[0].strip('.,;\'"`') 
            if url: # Ensure the stripped URL is not empty
                extracted_urls_set.add(url)
            
        result_urls = sorted(list(extracted_urls_set)) # Sort for consistent output

        logger.info(f"[{self.node_name}] Successfully extracted {len(result_urls)} unique URLs.")
        return result_urls