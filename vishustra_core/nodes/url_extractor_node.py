import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A specialized node for extracting web URLs from textual data.
    Uses a standard regex pattern to identify and isolate links for downstream tasks
    such as scraping, verification, or indexing.
    """

    # RFC 3986 compliant-ish regex for URL detection
    URL_REGEX = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "URL_Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Parses the input data to find all valid URLs.

        Args:
            data: The input payload, expected to be a string.
            context: Shared state/metadata for the current orchestration pipeline.

        Returns:
            List[str]: A list of unique URLs extracted from the input.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error("URLExtractorNode received non-string input of type: %s", type(data).__name__)
            raise TypeError(f"Expected string input for extraction, got {type(data).__name__}")

        try:
            logger.debug("Starting URL extraction on input data of length %d", len(data))
            
            # Find all matches
            raw_urls = self.URL_REGEX.findall(data)
            
            # Deduplicate while preserving order (Python 3.7+ dict behavior)
            unique_urls = list(dict.fromkeys(raw_urls))
            
            logger.info("Successfully extracted %d unique URLs.", len(unique_urls))
            return unique_urls

        except Exception as e:
            logger.error("An unexpected error occurred during URL extraction: %s", str(e), exc_info=True)
            raise RuntimeError("Internal node failure during URL extraction process.") from e

if __name__ == "__main__":
    # Internal component testing block
    extractor = URLExtractorNode()
    test_text = "Check out https://github.com/vishustra and also visit http://example.com/test?param=1"
    results = extractor.process(test_text, {})
    # Results handled by orchestration engine in production environment