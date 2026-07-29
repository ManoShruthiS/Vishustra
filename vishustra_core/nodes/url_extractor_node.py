import logging
import re
from typing import Any, Dict, List, Set, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts unique URLs from text content.
    It can process a single string or a list of strings, returning
    a unique list of discovered URLs.
    """

    # Compiled regular expression for robust URL matching.
    # This pattern aims to capture common URL formats, including those starting
    # with http(s)://, ftp://, or www., and covers various domain structures,
    # paths, queries, and fragments.
    _URL_REGEX = re.compile(
        r'(?:(?:https?|ftp):\/\/|www\.|ftp\.)'  # Protocol prefix (http, https, ftp) or www.
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+' # Domain name (e.g., example.com)
        r'[a-zA-Z]{2,6}'  # Top-level domain (e.g., com, org, net)
        r'(?::\d{1,5})?'  # Optional port number
        r'(?:[\/?#]\S*)?' # Optional path, query string, or fragment
        r'\b',            # Word boundary to ensure clean matches
        re.IGNORECASE
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        Args:
            data (Union[str, List[str]]): The input data to process.
                                         Expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for processing. Not directly used by this node,
                                      but required by the BaseNode interface.

        Returns:
            List[str]: A sorted list of unique URLs found in the input data.
                       Returns an empty list if no URLs are found or
                       if the input data is of an unsupported type.
        """
        extracted_urls: Set[str] = set()
        
        # Normalize input to always be an iterable of strings
        if isinstance(data, str):
            items_to_process = [data]
        elif isinstance(data, (list, tuple)):
            items_to_process = data
        else:
            logger.warning(
                f"[{self.node_name}] Received unsupported data type: {type(data).__name__}. "
                "Expected str or List[str]. Returning an empty list."
            )
            return []

        for item in items_to_process:
            if not isinstance(item, str):
                logger.debug(
                    f"[{self.node_name}] Skipping non-string item in input list: {type(item).__name__}. "
                    "Only string items are processed for URL extraction."
                )
                continue
            
            try:
                found_urls = self._URL_REGEX.findall(item)
                for url in found_urls:
                    # Normalize URLs that start with 'www.' but lack a protocol,
                    # by prepending 'http://' as a common default.
                    if url.startswith("www.") and not url.startswith(("http://", "https://", "ftp://")):
                        extracted_urls.add(f"http://{url}")
                    else:
                        extracted_urls.add(url)
                
                logger.debug(f"[{self.node_name}] Extracted {len(found_urls)} potential URLs from an item.")
            except re.error as e:
                logger.error(
                    f"[{self.node_name}] Regex error encountered while processing item (first 100 chars): '{item[:100]}...'. "
                    f"Error: {e}", exc_info=True
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred while processing item (first 100 chars): '{item[:100]}...'. "
                    f"Error: {e}", exc_info=True
                )

        result = sorted(list(extracted_urls))
        logger.debug(f"[{self.node_name}] Finished processing. Found {len(result)} unique URLs.")
        return result
