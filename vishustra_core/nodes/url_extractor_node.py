import logging
import re
from typing import Any, Dict, List, Union, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text input.
    It can process a single string or a list of strings, identifying
    standard HTTP/HTTPS URLs and returning them as a unique list.
    """

    # Comprehensive regular expression for URL matching, including common TLDs,
    # paths, query parameters, and fragments.
    # Ref: https://stackoverflow.com/questions/57106096/most-robust-url-regex-pattern
    _URL_PATTERN = re.compile(
        r'https?://(?:www\.)?'  # Optional 'www.'
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain name
        r'[a-zA-Z]{2,6}'  # TLD
        r'(?:/?|[/?]\S+)'  # Optional path, query or fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts all unique URLs from the input data.

        The input `data` can be a single string or a list of strings.
        Non-string elements in a list will be skipped with a warning.
        Returns a list of unique URLs found.

        Args:
            data: The input string or list of strings to process.
            context: A dictionary of contextual information (not used by this node).

        Returns:
            A list of unique URLs found in the input data.

        Raises:
            TypeError: If the input data is neither a string nor a list of strings.
            Exception: For any other unexpected errors during processing.
        """
        extracted_urls: Set[str] = set()

        if data is None:
            logger.warning(
                f"[{self.node_name}] Received None as input data. Returning empty list."
            )
            return []

        try:
            if isinstance(data, str):
                found_urls = self._URL_PATTERN.findall(data)
                extracted_urls.update(found_urls)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        found_urls = self._URL_PATTERN.findall(item)
                        extracted_urls.update(found_urls)
                    else:
                        logger.warning(
                            f"[{self.node_name}] Skipping non-string item in list: {type(item).__name__}. "
                            "Only strings are processed for URL extraction."
                        )
            else:
                raise TypeError(
                    f"[{self.node_name}] Invalid input data type. Expected str or List[str], "
                    f"but received {type(data).__name__}."
                )
        except TypeError as e:
            logger.error(f"[{self.node_name}] Input validation error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True
            )
            raise

        result = list(extracted_urls)
        logger.debug(f"[{self.node_name}] Extracted {len(result)} unique URLs.")
        return result

