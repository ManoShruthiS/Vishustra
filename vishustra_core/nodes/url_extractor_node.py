import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text content.

    This node takes a string as input data and returns a list of all
    identified URLs within that text. It supports common HTTP/HTTPS and
    'www.' prefixed URLs.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data (Any): The input data, expected to be a string containing text.
                        If a non-string type is provided, a TypeError will be raised.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by
                                       this specific node but required by the BaseNode interface.

        Returns:
            List[str]: A list of strings, where each string is an extracted URL.
                       Returns an empty list if no URLs are found.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name}: Invalid input data type. Expected a string, "
                f"but received {type(data).__name__}. Context: {context.get('task_id', 'N/A')}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.debug(f"{self.node_name}: Starting URL extraction for input data (length: {len(data)}).")

        # Regular expression to find URLs. This pattern targets common HTTP/HTTPS
        # URLs and those starting with 'www.'. It aims to be robust enough for
        # general text content without being overly permissive.
        # It captures:
        # - http:// or https:// followed by non-whitespace, non-angle bracket, non-quote characters.
        # - www. followed by non-whitespace, non-angle bracket, non-quote characters.
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

        extracted_urls = url_pattern.findall(data)

        if not extracted_urls:
            logger.info(f"{self.node_name}: No URLs found in the provided text.")
        else:
            logger.info(f"{self.node_name}: Successfully extracted {len(extracted_urls)} URL(s).")
            # For debugging, log the actual URLs, but keep info level lean
            logger.debug(f"{self.node_name}: Extracted URLs: {extracted_urls}")

        return extracted_urls