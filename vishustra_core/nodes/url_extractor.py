import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available from this path in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract unique URLs from string data.

    This node processes input text, identifies common HTTP/HTTPS URL patterns
    using regular expressions, and returns a sorted list of unique URLs found.
    It provides robust handling for non-string inputs and potential regex errors.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "URL_Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts all unique HTTP/HTTPS URLs from the input data.

        Args:
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted.
            context: A dictionary containing contextual information, which
                     is passed through but not directly used by this specific node.

        Returns:
            A sorted list of unique URLs (strings) found in the input data.
            Returns an empty list if no URLs are found, if the input data
            is not a string, or if an error occurs during extraction.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Received non-string data of type '%s'. Expected string for URL extraction. Returning empty list.",
                self.node_name,
                type(data).__name__
            )
            return []

        # Regular expression to find common HTTP/HTTPS URLs.
        # This pattern captures sequences starting with 'http://' or 'https://'
        # followed by one or more non-whitespace characters. This is a practical
        # balance for general URL extraction without excessive complexity.
        url_pattern = r"(https?://[^\s]+)"

        try:
            # Find all occurrences of the URL pattern in the data
            found_urls = re.findall(url_pattern, data)

            # Convert to a set to ensure uniqueness, then back to a sorted list
            unique_urls = sorted(list(set(found_urls)))

            if unique_urls:
                logger.debug(
                    "[%s] Successfully extracted %d unique URLs.",
                    self.node_name,
                    len(unique_urls)
                )
            else:
                logger.debug(
                    "[%s] No URLs found in the provided data.",
                    self.node_name
                )
            return unique_urls

        except re.error as e:
            # Catch specific regex compilation or execution errors
            logger.error(
                "[%s] A regex error occurred during URL extraction: %s",
                self.node_name,
                e
            )
            return []
        except Exception as e:
            # Catch any other unexpected errors during processing
            logger.error(
                "[%s] An unexpected error occurred during URL extraction: %s",
                self.node_name,
                e,
                exc_info=True # Log full traceback for debugging unexpected issues
            )
            return []