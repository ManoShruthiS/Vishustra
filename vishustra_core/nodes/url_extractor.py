import re
import logging
from typing import Any, Dict, List, Union, Set

# BaseNode is expected to be available at this path in the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from textual data.

    This node efficiently processes either a single string or a list of strings,
    identifying common URL patterns (e.g., those starting with http(s):// or www.)
    and returning a unique collection of the URLs found. It incorporates robust
    error handling for invalid input types, logging warnings and errors as appropriate.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all unique URLs.

        Args:
            data: The input, which can be a single string or a list of strings.
                  The node will iterate through strings to find URLs.
            context: A dictionary for contextual information; not utilized by this node
                     but included for `BaseNode` compatibility and future extensibility.

        Returns:
            A sorted list of unique URLs found in the input data. Returns an empty list
            if the input is invalid or if no URLs are detected.
        """
        all_extracted_urls: Set[str] = set()

        if data is None:
            logger.warning("URLExtractorNode received 'None' as input data. Returning an empty list of URLs.")
            return []

        # Ensure we have an iterable of strings to process
        texts_to_process: List[str]
        if isinstance(data, str):
            texts_to_process = [data]
        elif isinstance(data, list):
            texts_to_process = data
        else:
            logger.error(
                f"URLExtractorNode received an unsupported data type: {type(data).__name__}. "
                "Expected 'str' or 'List[str]'. Returning an empty list."
            )
            return []

        # Regex to capture common URL patterns.
        # This pattern matches URLs starting with 'http://', 'https://', or 'www.'
        # and continues to capture any non-whitespace, non-angle bracket, non-double quote characters.
        # This approach provides a good balance between generality and accuracy for typical text.
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

        for i, text in enumerate(texts_to_process):
            if not isinstance(text, str):
                logger.warning(
                    f"Skipping non-string item at index {i} in input list (type: {type(text).__name__}). "
                    "URLExtractorNode expects a list of strings."
                )
                continue
            
            # Find all matches in the current text block
            found_urls_in_text = url_pattern.findall(text)
            for url in found_urls_in_text:
                all_extracted_urls.add(url)
        
        # Convert the set to a sorted list for consistent and deterministic output.
        return sorted(list(all_extracted_urls))