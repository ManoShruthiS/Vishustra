import re
import logging
from typing import Any, Dict, List

# Assume vishustra_core.nodes.base_node exists and contains BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    It uses a regular expression to find common URL patterns, including
    http(s):// and www. prefixes, and returns a list of all unique URLs found.
    """

    _URL_PATTERN = re.compile(
        r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)',
        re.IGNORECASE
    )
    # A slightly broader pattern to also catch 'www.' without a scheme
    _BROADER_URL_PATTERN = re.compile(
        r'(?:https?:\/\/|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        re.IGNORECASE
    )


    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by this node.

        Returns:
            List[str]: A list of unique URLs found in the data. Returns an empty
                       list if no URLs are found or if the input data is not a string.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"URLExtractorNode expects input 'data' to be a string, but received type {type(data).__name__}."
            logger.error(error_msg, extra={"node_name": self.node_name, "input_type": type(data).__name__})
            raise TypeError(error_msg)

        if not data.strip():
            logger.debug("Input data is empty or whitespace only, no URLs to extract.", extra={"node_name": self.node_name})
            return []

        # Find all matches using the robust URL pattern
        urls_found = self._BROADER_URL_PATTERN.findall(data)

        # Ensure unique URLs and clean up potential trailing characters if any regex is too greedy
        # A set is used to easily handle uniqueness
        unique_urls = sorted(list(set(urls_found)))

        if not unique_urls:
            logger.debug("No URLs found in the provided text.", extra={"node_name": self.node_name})
        else:
            logger.info(f"Successfully extracted {len(unique_urls)} unique URLs.", extra={"node_name": self.node_name, "num_urls": len(unique_urls)})

        return unique_urls
