import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from text data.

    This node robustly identifies URLs starting with 'http://', 'https://',
    or 'www.' from single strings or lists of strings. It handles various common
    URL formats and ensures uniqueness and cleanliness of the extracted URLs.
    """

    # Regex pattern to find URLs.
    # It looks for URLs starting with 'http(s)://' or 'www.'
    # followed by any sequence of non-whitespace characters (`\S+`).
    # The `\b` word boundary at the end helps to prevent capturing unwanted
    # trailing characters (e.g., spaces or certain punctuation) that might
    # immediately follow a URL. The `(?i)` flag makes the pattern case-insensitive.
    _URL_PATTERN = re.compile(
        r'(?i)\b(?:https?://|www\.)\S+\b'
    )
    
    # Common trailing punctuation characters that are often captured by the broad
    # `\S+` pattern but are typically not an intrinsic part of a URL and should
    # be stripped for cleaner output.
    _TRAILING_PUNCTUATION = '.,;:)]}\'\"'

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URLExtractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the provided input data.

        Args:
            data: The input data, expected to be either a single string or a
                  list of strings. Non-string items within a list will be
                  skipped and a warning will be logged.
            context: A dictionary containing contextual information relevant
                     to the processing task, such as a 'task_id' for logging.

        Returns:
            A list of unique and cleaned URLs found in the input data,
            sorted alphabetically for consistent output. Returns an empty list
            if no URLs are found or if the input data type is unsupported.
        """
        extracted_urls: set[str] = set()
        task_id = context.get('task_id', 'N/A')
        
        text_inputs: List[str] = []
        if isinstance(data, str):
            text_inputs.append(data)
        elif isinstance(data, list):
            # Filter the list to include only string items, logging any non-string elements.
            non_string_items_count = sum(1 for item in data if not isinstance(item, str))
            if non_string_items_count > 0:
                logger.warning(
                    f"[{self.node_name}] Received a list containing {non_string_items_count} "
                    f"non-string item(s). Processing only string elements. Task ID: {task_id}"
                )
            text_inputs.extend([item for item in data if isinstance(item, str)])
        else:
            logger.warning(
                f"[{self.node_name}] Received unsupported data type: '{type(data).__name__}'. "
                f"Expected 'str' or 'List[str]'. Returning empty list. Task ID: {task_id}"
            )
            return []

        # Iterate through all valid text inputs to find and clean URLs.
        for text in text_inputs:
            found_urls = self._URL_PATTERN.findall(text)
            for url in found_urls:
                # Strip common trailing punctuation characters that are often
                # captured by the broad `\S+` pattern but are not part of the actual URL.
                cleaned_url = url.rstrip(self._TRAILING_PUNCTUATION)
                
                # Add the cleaned URL to the set if it's not empty after stripping.
                # The regex is designed to avoid matching just "http://" or "www."
                # without additional non-whitespace characters.
                if cleaned_url:
                    extracted_urls.add(cleaned_url)

        # Convert the set of unique URLs to a sorted list for consistent output.
        result_list = sorted(list(extracted_urls))
        
        if not result_list:
            logger.debug(
                f"[{self.node_name}] Found no URLs in the input data. Task ID: {task_id}"
            )
        else:
            logger.debug(
                f"[{self.node_name}] Extracted {len(result_list)} unique URL(s): {result_list}. Task ID: {task_id}"
            )
        
        return result_list
