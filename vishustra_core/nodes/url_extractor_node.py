
import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from text data.

    This node processes input data (either a single string or a list of strings)
    to identify and extract all unique URLs present within the text content.
    It leverages a robust regular expression to accurately capture various URL formats.
    """

    # A regular expression pattern to identify URLs.
    # It covers schemes like http/https and common patterns starting with 'www.',
    # followed by domain names and optional paths/queries/fragments.
    # This regex aims for broad coverage of valid URL characters.
    URL_REGEX = re.compile(
        r'(?:https?://|www\.)'  # Match 'http://', 'https://', or 'www.'
        r'(?:[a-zA-Z0-9-]+\.)+'  # Match domain parts (e.g., example.com)
        r'[a-zA-Z]{2,6}'         # Match top-level domain (e.g., com, org, co.uk)
        r'(?:/[^\s]*)?'          # Optionally match path, query, fragment (anything non-space)
        r'(?=[.,!?;\])}"\s]|$)'  # Positive lookahead to ensure URL ends cleanly or with whitespace/end of string
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "URLExtractorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts all unique URLs from the input data.

        The `data` parameter can be a single string or a list of strings.
        The node iterates through the text content, applies a regular expression
        to find potential URLs, and compiles them into a unique list.

        Args:
            data: The input text data from which URLs are to be extracted.
                  Can be a `str` or a `List[str]`.
            context: A dictionary containing contextual information for processing.
                     This node does not currently utilize the context, but it's
                     part of the `BaseNode` interface.

        Returns:
            A `List[str]` containing all unique URLs found in the input data.
            Returns an empty list if no URLs are found or if the input data
            is invalid or empty.
        """
        if not data:
            logger.debug(f"{self.node_name}: Received empty or None data. Returning an empty list.")
            return []

        all_extracted_urls = set()

        if isinstance(data, str):
            logger.debug(f"{self.node_name}: Processing a single string input.")
            extracted = self._extract_from_text(data)
            all_extracted_urls.update(extracted)
        elif isinstance(data, list):
            logger.debug(f"{self.node_name}: Processing a list of strings input.")
            for item in data:
                if isinstance(item, str):
                    extracted = self._extract_from_text(item)
                    all_extracted_urls.update(extracted)
                else:
                    logger.warning(
                        f"{self.node_name}: Encountered a non-string item in the input list "
                        f"(type: {type(item).__name__}). Skipping for URL extraction."
                    )
        else:
            logger.error(
                f"{self.node_name}: Invalid input data type. Expected 'str' or 'List[str]', "
                f"but received '{type(data).__name__}'. Returning an empty list."
            )
            return []

        result_urls = sorted(list(all_extracted_urls))

        if result_urls:
            logger.info(f"{self.node_name}: Successfully extracted {len(result_urls)} unique URLs.")
            logger.debug(f"{self.node_name}: Extracted URLs: {result_urls}")
        else:
            logger.info(f"{self.node_name}: No URLs were found in the provided data.")

        return result_urls

    def _extract_from_text(self, text: str) -> List[str]:
        """
        Helper method to apply the URL regular expression to a single string.

        Args:
            text: The string from which to extract URLs.

        Returns:
            A list of URLs found in the text.
        """
        found_urls = self.URL_REGEX.findall(text)
        logger.debug(f"{self.node_name}: Found {len(found_urls)} URLs in a text segment.")
        return found_urls
