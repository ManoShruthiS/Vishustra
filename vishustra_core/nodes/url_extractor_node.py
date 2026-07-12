import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from input text.

    This node identifies common HTTP/HTTPS URL patterns within the input data.
    It can process a single string or a list of strings.
    """

    # A robust regex compiled for efficiency, designed to capture common HTTP/HTTPS URLs.
    # It covers scheme, host, path, query parameters, and common special characters
    # allowed in URLs.
    _URL_REGEX = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Helper method to extract URLs from a single string using the compiled regex.
        """
        try:
            return self._URL_REGEX.findall(text)
        except TypeError as e:
            logger.error(f"[{self.node_name}] TypeError during URL extraction from text: {e}")
            return []
        except Exception as e:
            logger.error(f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}")
            return []

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract all unique URLs.

        The node expects input data to be either a single string or a list of strings.
        It uses a regular expression to find HTTP/HTTPS URLs. If the input data type
        is not supported, an error is logged, and an empty list is returned.
        Non-string elements within an input list are skipped with a warning.

        Args:
            data: The input data, expected to be a `str` or `List[str]`.
            context: A dictionary for shared pipeline context. This node does not
                     currently utilize the context.

        Returns:
            A sorted list of unique URLs found in the input data. Returns an empty
            list if no URLs are found or if the input data type is unsupported.
        """
        all_extracted_urls: List[str] = []

        if isinstance(data, str):
            urls = self._extract_urls_from_text(data)
            all_extracted_urls.extend(urls)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    urls = self._extract_urls_from_text(item)
                    all_extracted_urls.extend(urls)
                else:
                    logger.warning(
                        f"[{self.node_name}] Skipping non-string item at index {i} in input list. "
                        f"Expected `str`, got `{type(item).__name__}`."
                    )
        else:
            logger.error(
                f"[{self.node_name}] Unsupported input data type: `{type(data).__name__}`. "
                "Expected `str` or `List[str]`."
            )
            # For unsupported types, return an empty list to allow pipeline continuation
            return []

        # Convert to a set to get unique URLs, then back to a sorted list for consistent output.
        unique_urls = list(set(all_extracted_urls))
        return sorted(unique_urls)