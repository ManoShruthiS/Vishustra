import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists and contains the BaseNode definition.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from text data.

    This node is designed to identify and extract URLs from input strings
    or lists of strings, providing a consolidated list of unique URLs.
    It uses a robust regular expression pattern to maximize coverage of
    common URL formats while minimizing false positives.
    """

    # A comprehensive regular expression pattern for URL extraction.
    # This pattern is designed to capture standard HTTP/HTTPS URLs,
    # including those starting with 'www.', and properly handle
    # various path segments, query parameters, and fragment identifiers.
    # It also accounts for common non-URL punctuation that might trail a URL.
    # Source: Adapted from common robust URL regex patterns, e.g., for linkifiers.
    _URL_PATTERN = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'  # Scheme or domain start
        r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+' # Main URL content, handling nested parentheses
        r'(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))' # Trailing char handling
    )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all unique URLs.

        The input `data` can be a single string or a list of strings.
        The `context` dictionary can be utilized for future enhancements,
        such as configurable regex patterns or extraction options.

        Args:
            data: The input text content, expected as a `str` or `List[str]`.
            context: A dictionary containing execution context or configurations.
                     Currently, this node does not leverage context for its core logic,
                     but it's available for extensions.

        Returns:
            A `List[str]` containing all unique URLs found in the input data.
            Returns an empty list if no URLs are found, or if the input data
            is not of a supported type.
        """
        extracted_urls: List[str] = []
        
        logger.debug(f"URLExtractorNode received context for process: {context}")

        if isinstance(data, str):
            text_inputs = [data]
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            text_inputs = data
        else:
            logger.warning(
                f"URLExtractorNode received unsupported data type: {type(data)}. "
                "Expected `str` or `List[str]`. Returning an empty list."
            )
            return []

        for text in text_inputs:
            if not text: # Skip empty strings
                continue
            try:
                # Use the pre-compiled regex pattern for efficient searching
                found_urls = self._URL_PATTERN.findall(text)
                
                # The regex might return tuples for groups; flatten them
                # and clean up empty strings that might arise from optional groups.
                cleaned_urls = [
                    url for match_tuple in found_urls 
                    for url in match_tuple if isinstance(url, str) and url
                ]
                
                if cleaned_urls:
                    extracted_urls.extend(cleaned_urls)
                    logger.debug(f"Extracted {len(cleaned_urls)} URLs from a text segment (first 50 chars: '{text[:50]}').")
            except Exception as e:
                # Log the error but continue processing other text segments
                logger.error(
                    f"Error extracting URLs from text segment (first 100 chars: '{text[:100]}...'). "
                    f"Error: {e}", exc_info=True
                )

        # Remove duplicate URLs while preserving insertion order for consistency
        unique_urls = list(dict.fromkeys(extracted_urls))
        logger.info(f"URL extraction complete. Found {len(unique_urls)} unique URLs.")
        
        return unique_urls