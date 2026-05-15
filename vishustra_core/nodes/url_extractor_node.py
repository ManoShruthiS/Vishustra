import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A specialized node for identifying and extracting URLs from unstructured text data.
    Uses regular expressions to find all HTTP/HTTPS links within the provided input.
    """

    # RFC 3986 compliant-ish regex for standard URL extraction
    _URL_REGEX = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node type."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Scans the input data for URLs.

        Args:
            data (Any): The input data to process. Expected to be a string.
            context (Dict[str, Any]): Pipeline context containing metadata or configurations.

        Returns:
            List[str]: A list of unique URLs extracted from the input text.

        Raises:
            TypeError: If the input data is not a string.
            Exception: For unexpected processing failures.
        """
        if not isinstance(data, str):
            error_msg = f"Node '{self.node_name}' expected string input, but received {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            logger.debug(f"Starting URL extraction on input of length {len(data)}.")
            
            # Find all matches
            found_urls = self._URL_REGEX.findall(data)
            
            # Deduplicate while preserving order (using dict keys for older Python compatibility/stability)
            unique_urls = list(dict.fromkeys(found_urls))
            
            logger.info(f"Extracted {len(unique_urls)} unique URLs.")
            return unique_urls

        except Exception as e:
            logger.error(f"Failed to extract URLs in {self.node_name}: {str(e)}", exc_info=True)
            raise Exception(f"URL extraction failed: {str(e)}") from e

if __name__ == "__main__":
    # Internal module testing logic if executed directly
    logging.basicConfig(level=logging.INFO)
    test_node = URLExtractorNode()
    sample_text = "Visit our site at https://example.com or check our docs at http://docs.vishustra.io."
    results = test_node.process(sample_text, {})
    logger.info(f"Test Result: {results}")

