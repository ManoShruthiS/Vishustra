import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is located in this relative path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from textual content.

    This node takes either a single string or a list of strings as input. It
    scans the provided text(s) for patterns that match common URL structures
    (e.g., starting with http://, https://, or www.) and returns a list of all
    unique URLs found.
    """

    # A robust regex pattern for identifying URLs in text.
    # It covers common schemes (http/https/www.) and captures subsequent
    # non-whitespace characters until a word boundary or whitespace is encountered.
    _URL_REGEX = re.compile(
        r'\b(?:https?://|www\.)'  # Match 'http://', 'https://', or 'www.' at a word boundary
        r'[^\s/$.?#].[^\s]*'      # Capture domain, path, query, etc. (any non-whitespace character,
                                  # ensuring at least one character follows the prefix,
                                  # then any number of non-whitespace characters)
        r'\b'                     # End at a word boundary to prevent partial matches within words
    )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "URLExtractorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        The input `data` can be a single string or a list of strings. Each string
        is parsed to find URL patterns. The `context` dictionary is provided but
        is not directly utilized by this node's core logic, allowing for future
        configuration extensions.

        Args:
            data: The input text(s) from which URLs need to be extracted.
                  Can be a `str` or `List[str]`.
            context: A dictionary containing runtime contextual information.

        Returns:
            A list of unique URLs found in the input data. The list will be empty
            if no URLs are found, or if the input data is invalid (after logging warnings).

        Raises:
            TypeError: If the input 'data' is neither a string nor a list of strings.
        """
        if data is None:
            logger.warning(f"[{self.node_name}] Received None as input data. Returning an empty list of URLs.")
            return []

        all_extracted_urls: List[str] = []

        if isinstance(data, str):
            all_extracted_urls.extend(self._extract_from_single_text(data))
        elif isinstance(data, list):
            for index, item in enumerate(data):
                if isinstance(item, str):
                    all_extracted_urls.extend(self._extract_from_single_text(item))
                else:
                    logger.warning(
                        f"[{self.node_name}] Encountered non-string item at index {index} in list input "
                        f"(type: {type(item).__name__}). Skipping this item."
                    )
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data).__name__}. "
                "Expected 'str' or 'List[str]'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string or a list of strings, "
                f"but received type: {type(data).__name__}."
            )

        # Remove duplicate URLs while preserving the order of first appearance.
        # dict.fromkeys() is used for this purpose (Python 3.7+).
        unique_urls = list(dict.fromkeys(all_extracted_urls))
        logger.info(f"[{self.node_name}] Successfully extracted {len(unique_urls)} unique URLs.")
        return unique_urls

    def _extract_from_single_text(self, text: str) -> List[str]:
        """
        Helper method to perform URL extraction from a single string using the
        pre-compiled regex.
        """
        if not text:
            return []
        return self._URL_REGEX.findall(text)