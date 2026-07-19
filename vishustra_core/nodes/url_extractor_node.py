import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text content.

    This node can process either a single string or a list of strings,
    identifying common URL patterns including `http(s)://` and `www.` prefixes.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URLExtractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Expected input:
            - A single string: The node will extract all URLs from this string.
            - A list of strings: The node will iterate through the list, extracting
              URLs from each string and aggregating them. Non-string items in the
              list will be logged as warnings and skipped.

        Args:
            data: The input text content(s) from which to extract URLs.
            context: A dictionary containing contextual information. This node
                     does not utilize the context for its core functionality.

        Returns:
            A list of unique URLs found in the input data, preserving their
            original order of appearance as much as possible.

        Raises:
            TypeError: If the input 'data' is neither a string nor a list of strings.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"URLExtractorNode received invalid data type. Expected 'str' or 'list[str]', "
                f"but got '{type(data).__name__}'."
            )
            raise TypeError("URLExtractorNode expects 'data' to be a string or a list of strings.")

        all_urls: List[str] = []

        # Robust regex pattern to identify common URL formats:
        # - Matches URLs starting with http:// or https://
        # - Matches URLs starting with www.
        # It captures non-whitespace, non-quote, non-angle bracket characters following these prefixes.
        # This is a pragmatic balance for general text extraction.
        url_pattern = re.compile(
            r'https?://[^\s<>"]+|www\.[^\s<>"]+',
            re.IGNORECASE  # Case-insensitive for 'http', 'www', etc.
        )

        if isinstance(data, str):
            found_urls = url_pattern.findall(data)
            all_urls.extend(found_urls)
            logger.debug(f"Extracted {len(found_urls)} URLs from single string input.")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    found_urls = url_pattern.findall(item)
                    all_urls.extend(found_urls)
                    logger.debug(f"Extracted {len(found_urls)} URLs from item at index {i}.")
                else:
                    logger.warning(
                        f"URLExtractorNode skipped non-string item at index {i} in input list. "
                        f"Expected 'str', but got '{type(item).__name__}'."
                    )
        
        # Deduplicate URLs while preserving order of first appearance.
        unique_urls = list(dict.fromkeys(all_urls))
        
        if not unique_urls:
            logger.info("URLExtractorNode completed, but no URLs were found in the provided data.")
        else:
            logger.info(f"URLExtractorNode successfully extracted {len(unique_urls)} unique URLs.")
            
        return unique_urls