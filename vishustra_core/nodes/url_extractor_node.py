import re
import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    It expects a string as input and returns a list of all unique URLs found
    within the text. URLs are identified using a robust regular expression.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data (Any): The input data, expected to be a string containing text.
                        If not a string, a warning is logged, and an empty list is returned.
            context (Dict[str, Any]): A dictionary containing context information
                                     for the current processing flow. Not directly used
                                     by this node, but passed along.

        Returns:
            List[str]: A list of unique URLs found in the input text. Returns an
                       empty list if no URLs are found or if the input data is
                       not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning an empty list."
            )
            return []

        # A comprehensive regex to capture various URL formats, including
        # http(s)://, www., and common TLDs, IP addresses, and paths.
        # This regex attempts to be robust without being overly permissive.
        url_pattern = re.compile(
            r'\b(?:https?://|www\.)'  # Scheme or www.
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # Domain name
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
            r'(?::\d+)?'  # Optional port
            r'(?:/?|[/?]\S+)\b',  # Path, query string and fragment
            re.IGNORECASE
        )

        try:
            found_urls = url_pattern.findall(data)
            unique_urls = sorted(list(set(found_urls))) # Remove duplicates and sort for consistency

            if not unique_urls:
                logger.debug(f"[{self.node_name}] No URLs found in the provided text.")
            else:
                logger.info(f"[{self.node_name}] Successfully extracted {len(unique_urls)} unique URLs.")
                logger.debug(f"[{self.node_name}] Extracted URLs: {unique_urls}")

            return unique_urls
        except re.error as e:
            logger.error(f"[{self.node_name}] Regular expression error during URL extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}", exc_info=True)
            return []

```
```python
# Helper/Mock for local testing (not part of the submission, but useful for verification)
if __name__ == '__main__':
    # Mock BaseNode for standalone execution context
    from abc import ABC, abstractmethod
    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass
    # Re-import URLExtractorNode after mocking BaseNode to pick up the local definition
    # This is a hack for local testing, not how actual imports work in a real project
    # In a real project, vishustra_core.nodes.base_node would be available.
    
    # For a clean test, we redefine the class here or mock the module structure
    # Let's mock the module structure slightly differently for a cleaner standalone test.

    import sys
    from unittest.mock import Mock
    # Create a mock module path
    class MockBaseNodeModule:
        BaseNode = BaseNode # Use the locally defined mock BaseNode

    sys.modules['vishustra_core.nodes.base_node'] = MockBaseNodeModule()

    # Now, re-run the user's provided code to get the URLExtractorNode
    # This is just for local testing simulation
    # The actual submission would only contain the URLExtractorNode class and its imports
    
    # Let's assume the URLExtractorNode class is available directly for testing purposes.
    # In a real scenario, you'd import it after the mock setup:
    # from url_extractor_node import URLExtractorNode 
    # For this simple test, I'll just copy-paste the class content for immediate execution
    
    # Re-declare URLExtractorNode here for self-contained testing, if not importing directly
    # from above.
    # However, the user's prompt demands a file, so I'll trust the main block above is correct
    # and this part is just for my internal validation.
    
    # Configure logging for better visibility during testing
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    node = URLExtractorNode()
    
    print(f"\n--- Testing Node: {node.node_name} ---")

    test_data_1 = "Check out my website at https://www.example.com/path?query=123 and also visit http://test.org. And here is another one: example.co.uk and an IP: http://192.168.1.1/admin"
    print(f"\nTest 1 (Valid URLs):\nInput: {test_data_1}")
    urls_1 = node.process(test_data_1, {})
    print(f"Extracted URLs: {urls_1}")
    expected_urls_1 = ['http://192.168.1.1/admin', 'http://test.org', 'https://www.example.com/path?query=123'] # Sorted
    assert sorted(urls_1) == sorted(expected_urls_1)
    
    test_data_2 = "No URLs here, just some plain text."
    print(f"\nTest 2 (No URLs):\nInput: {test_data_2}")
    urls_2 = node.process(test_data_2, {})
    print(f"Extracted URLs: {urls_2}")
    assert urls_2 == []

    test_data_3 = "Another one: www.google.com/search?q=python. And a duplicate: https://www.google.com/search?q=python"
    print(f"\nTest 3 (Duplicate URLs):\nInput: {test_data_3}")
    urls_3 = node.process(test_data_3, {})
    print(f"Extracted URLs: {urls_3}")
    assert urls_3 == ['https://www.google.com/search?q=python'] # Set removes duplicates

    test_data_4 = 12345
    print(f"\nTest 4 (Invalid Input - int):\nInput: {test_data_4}")
    urls_4 = node.process(test_data_4, {})
    print(f"Extracted URLs: {urls_4}")
    assert urls_4 == []

    test_data_5 = None
    print(f"\nTest 5 (Invalid Input - None):\nInput: {test_data_5}")
    urls_5 = node.process(test_data_5, {})
    print(f"Extracted URLs: {urls_5}")
    assert urls_5 == []

    test_data_6 = "ftp://ftp.example.com and file:///home/user/document.pdf are not covered by this regex."
    print(f"\nTest 6 (Non-HTTP/HTTPS/WWW URLs):\nInput: {test_data_6}")
    urls_6 = node.process(test_data_6, {})
    print(f"Extracted URLs: {urls_6}")
    assert urls_6 == [] # Expected not to find these with the current regex

    print("\nAll tests passed!")

