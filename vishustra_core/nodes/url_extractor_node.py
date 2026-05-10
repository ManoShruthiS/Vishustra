import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for the module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A specialized node designed to identify and extract HTTP/HTTPS URLs from raw text data.
    This node is useful for pre-processing steps where downstream nodes require 
    specific endpoints for scraping or analysis.
    """

    # RFC 3986 compliant-ish regex for URL discovery
    URL_PATTERN = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    @property
    def node_name(self) -> str:
        """
        Returns the human-readable identifier for this node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Parses the input data to extract all unique URLs.

        Args:
            data (Any): The input payload, expected to be a string.
            context (Dict[str, Any]): Global context dictionary for the pipeline execution.

        Returns:
            List[str]: A list of unique URLs found in the input data.

        Raises:
            TypeError: If the input data is not a string.
            Exception: For unexpected regex processing errors.
        """
        logger.debug(f"Executing node: {self.node_name}")

        if not isinstance(data, str):
            error_msg = f"URLExtractorNode expected string input, but received: {type(data).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Extract all matches using the predefined pattern
            extracted_urls = re.findall(self.URL_PATTERN, data)
            
            # De-duplicate while preserving original order
            unique_urls = list(dict.fromkeys(extracted_urls))
            
            logger.info(f"Extraction complete. Found {len(unique_urls)} unique URLs.")
            
            return unique_urls

        except re.error as e:
            logger.error(f"Regex compilation or execution error: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"An unhandled exception occurred during URL extraction: {str(e)}")
            raise

if __name__ == "__main__":
    # Internal dev testing - Node should be typically invoked via the Vishustra runner
    pass