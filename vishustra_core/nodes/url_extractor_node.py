import re
import logging
from typing import Any, Dict, List, Union

# Assume vishustra_core.nodes.base_node is available and contains BaseNode
# For local testing, you might need a placeholder or actual structure.
# For this submission, we strictly follow the import instruction.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from input text data.

    It can process a single string or a list of strings, returning a list
    of all unique URLs found.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        The input `data` can be:
        - A single string: URLs are extracted from this string.
        - A list of strings: URLs are extracted from each string in the list.

        Args:
            data (Union[str, List[str]]): The input text data to process.
            context (Dict[str, Any]): A dictionary for additional context or
                                       shared state across nodes (not used here).

        Returns:
            List[str]: A list of unique URLs found in the input data.
                       Returns an empty list if no URLs are found or if
                       the input data type is invalid.
        """
        all_urls: List[str] = []

        # Regex for URLs: A robust pattern that covers http/https, optional www,
        # domain, path, query parameters, and fragments.
        # This regex broadly matches common URL structures.
        url_pattern = r'https?:\/\/(?:www\.)?[a-zA-Z0-9-]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'

        if isinstance(data, str):
            urls = re.findall(url_pattern, data)
            all_urls.extend(urls)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    urls = re.findall(url_pattern, item)
                    all_urls.extend(urls)
                else:
                    logger.warning(
                        f"Skipping non-string item in list for URL extraction: {type(item)}. "
                        "Expected string."
                    )
        else:
            logger.warning(
                f"Invalid data type for URLExtractorNode. Expected str or List[str], "
                f"but received {type(data)}. Returning empty list."
            )
            return []

        # Return unique URLs to avoid duplicates if the same URL appears multiple times
        # in the input or across different strings in a list.
        return sorted(list(set(all_urls)))

if __name__ == '__main__':
    # This block is for local testing and demonstration purposes only.
    # In a real Vishustra setup, nodes are orchestrated differently.
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    # Placeholder for BaseNode if running standalone
    if 'BaseNode' not in locals():
        from abc import ABC, abstractmethod
        class BaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
            @property
            @abstractmethod
            def node_name(self) -> str: pass

    node = URLExtractorNode()
    dummy_context = {}

    print(f"--- Testing {node.node_name} ---")

    # Test Case 1: Single string with multiple URLs
    text1 = "Visit our website at https://www.example.com/page?id=123 or check out our blog at http://blog.anothersite.org/post#section. Also a short one: https://short.ly"
    extracted_urls = node.process(text1, dummy_context)
    print(f"Input (str): '{text1[:70]}...'")
    print(f"Extracted URLs: {extracted_urls}")
    assert "https://www.example.com/page?id=123" in extracted_urls
    assert "http://blog.anothersite.org/post#section" in extracted_urls
    assert "https://short.ly" in extracted_urls
    assert len(extracted_urls) == 3

    print("-" * 20)

    # Test Case 2: List of strings
    texts2 = [
        "Find more info at https://docs.project.io/v1/api.",
        "Our main page is https://project.io, and support is via https://support.project.io/tickets.",
        "No URLs here."
    ]
    extracted_urls = node.process(texts2, dummy_context)
    print(f"Input (list[str]): {texts2}")
    print(f"Extracted URLs: {extracted_urls}")
    assert "https://docs.project.io/v1/api" in extracted_urls
    assert "https://project.io" in extracted_urls
    assert "https://support.project.io/tickets" in extracted_urls
    assert len(extracted_urls) == 3

    print("-" * 20)

    # Test Case 3: No URLs
    text3 = "This is some plain text without any links whatsoever."
    extracted_urls = node.process(text3, dummy_context)
    print(f"Input (str, no URLs): '{text3}'")
    print(f"Extracted URLs: {extracted_urls}")
    assert len(extracted_urls) == 0

    print("-" * 20)

    # Test Case 4: Invalid input type
    invalid_data = 12345
    extracted_urls = node.process(invalid_data, dummy_context)
    print(f"Input (invalid type): {invalid_data}")
    print(f"Extracted URLs: {extracted_urls}")
    assert len(extracted_urls) == 0

    print("-" * 20)

    # Test Case 5: List with mixed valid/invalid types
    mixed_list = ["URL: https://mixed.com", 123, "Another one: http://example.net"]
    extracted_urls = node.process(mixed_list, dummy_context)
    print(f"Input (mixed list): {mixed_list}")
    print(f"Extracted URLs: {extracted_urls}")
    assert "https://mixed.com" in extracted_urls
    assert "http://example.net" in extracted_urls
    assert len(extracted_urls) == 2

    print("\nAll tests passed successfully for URLExtractorNode!")