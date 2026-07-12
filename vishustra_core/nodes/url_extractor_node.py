import re
import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists at this path relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node uses a regular expression to find common URL patterns within
    the input string. It supports HTTP, HTTPS protocols, and optionally
    www. prefixes.
    """

    # A robust regex for common URLs, including schemes, domains, paths,
    # queries, and fragments.
    _URL_REGEX = re.compile(
        r'https?:\/\/(?:www\.)?'  # Scheme (http/https) and optional www.
        r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.'  # Domain name
        r'[a-zA-Z0-9()]{1,6}\b'  # Top-level domain
        r'(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)' # Optional path, query, fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL_Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all URLs.

        Args:
            data (Any): The input data, expected to be a string containing text.
                        If not a string, an empty list will be returned and a
                        warning logged.
            context (Dict[str, Any]): A dictionary containing context information
                                     for the current processing flow. Not directly
                                     used by this node but passed along.

        Returns:
            List[str]: A list of unique URLs found in the input data.
                       Returns an empty list if no URLs are found or if the
                       input data is invalid.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type for data. Expected str, got {type(data)}. "
                "Returning an empty list of URLs."
            )
            return []

        if not data.strip():
            logger.debug(f"[{self.node_name}] Received empty or whitespace-only data. Returning an empty list.")
            return []

        try:
            found_urls = self._URL_REGEX.findall(data)
            unique_urls = list(dict.fromkeys(found_urls)) # Preserve order while making unique

            if unique_urls:
                logger.info(f"[{self.node_name}] Successfully extracted {len(unique_urls)} unique URLs.")
            else:
                logger.debug(f"[{self.node_name}] No URLs found in the provided data.")

            return unique_urls
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            return []

# Example of how to use this node (for internal testing/understanding, not part of the delivered code)
if __name__ == "__main__":
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock BaseNode path for local execution
    import sys
    import os
    # Create a dummy vishustra_core/nodes directory structure
    os.makedirs('vishustra_core/nodes', exist_ok=True)
    with open('vishustra_core/nodes/base_node.py', 'w') as f:
        f.write("""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseNode(ABC):
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        pass
    
    @property
    @abstractmethod
    def node_name(self) -> str:
        pass
""")
    # Add current directory to path to allow import
    sys.path.insert(0, os.getcwd())

    # Create an instance of the node
    url_extractor = URLExtractorNode()

    # Test cases
    test_data_1 = "Check out our website at https://www.example.com and also visit http://test.org/path?id=123. For more info: example.com/not-a-full-url"
    test_data_2 = "No URLs here, just plain text."
    test_data_3 = "Multiple links to https://google.com and then again to https://google.com/search?q=test and a new one https://example.net"
    test_data_4 = 12345  # Invalid input type
    test_data_5 = ""     # Empty string
    test_data_6 = "   "  # Whitespace only

    print(f"\n--- Testing {url_extractor.node_name} ---")

    print(f"\nProcessing data 1: '{test_data_1}'")
    extracted_urls_1 = url_extractor.process(test_data_1, {})
    print(f"Extracted URLs 1: {extracted_urls_1}") # Expected: ['https://www.example.com', 'http://test.org/path?id=123']

    print(f"\nProcessing data 2: '{test_data_2}'")
    extracted_urls_2 = url_extractor.process(test_data_2, {})
    print(f"Extracted URLs 2: {extracted_urls_2}") # Expected: []

    print(f"\nProcessing data 3: '{test_data_3}'")
    extracted_urls_3 = url_extractor.process(test_data_3, {})
    print(f"Extracted URLs 3: {extracted_urls_3}") # Expected: ['https://google.com', 'https://google.com/search?q=test', 'https://example.net']

    print(f"\nProcessing data 4 (invalid type): '{test_data_4}'")
    extracted_urls_4 = url_extractor.process(test_data_4, {})
    print(f"Extracted URLs 4: {extracted_urls_4}") # Expected: [] and a warning log

    print(f"\nProcessing data 5 (empty string): '{test_data_5}'")
    extracted_urls_5 = url_extractor.process(test_data_5, {})
    print(f"Extracted URLs 5: {extracted_urls_5}") # Expected: [] and a debug log

    print(f"\nProcessing data 6 (whitespace only): '{test_data_6}'")
    extracted_urls_6 = url_extractor.process(test_data_6, {})
    print(f"Extracted URLs 6: {extracted_urls_6}") # Expected: [] and a debug log

    # Clean up dummy base_node files
    sys.path.pop(0)
    os.remove('vishustra_core/nodes/base_node.py')
    os.rmdir('vishustra_core/nodes')
    os.rmdir('vishustra_core')
