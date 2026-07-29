import re
import logging
from typing import Any, Dict, List, Union, Set

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node can process a single string or a list of strings, identifying
    common HTTP/HTTPS URLs and returning them as a unique, sorted list.
    """

    # A robust regular expression pattern for matching common HTTP/HTTPS URLs.
    # This pattern covers common domain structures, optional 'www.',
    # and potential paths, queries, or fragments.
    _url_regex_pattern: str = r'https?://(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+(?:/[^"\s]*)?'

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URLExtractorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        The input `data` can be a single string or a list of strings.
        URLs are identified using a robust regular expression. The `context`
        parameter is currently not utilized by this node but is provided
        for future extensibility.

        Args:
            data: The input text (string or list of strings) from which to extract URLs.
            context: A dictionary containing contextual information for processing.

        Returns:
            A list of unique URLs found in the input data, sorted alphabetically.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
        """
        extracted_urls: Set[str] = set()

        if isinstance(data, str):
            urls_in_data = re.findall(self._url_regex_pattern, data)
            extracted_urls.update(urls_in_data)
            logger.debug(f"Processed single string input. Found {len(urls_in_data)} URLs.")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    urls_in_item = re.findall(self._url_regex_pattern, item)
                    extracted_urls.update(urls_in_item)
                    logger.debug(f"Processed list item {i}. Found {len(urls_in_item)} URLs.")
                else:
                    logger.warning(
                        f"URLExtractorNode received a non-string item at index {i} in the input list "
                        f"(type: {type(item).__name__}). Skipping this item."
                    )
        else:
            logger.error(
                f"Invalid input data type for URLExtractorNode. "
                f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'."
            )
            raise TypeError("URLExtractorNode expects 'data' to be a string or a list of strings.")

        result = sorted(list(extracted_urls))

        if not result:
            logger.info("URLExtractorNode finished processing, but no URLs were found.")
        else:
            logger.info(f"URLExtractorNode finished processing. Extracted {len(result)} unique URLs.")
            logger.debug(f"Extracted URLs: {result}")

        return result
