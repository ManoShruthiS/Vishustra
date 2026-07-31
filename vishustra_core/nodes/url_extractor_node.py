import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from textual data.

    This node processes an input string, identifying and returning all URLs
    that conform to common web address patterns (e.g., http(s):// or www.).
    It provides a reliable way to isolate web links for further processing
    or analysis within an orchestration flow.
    """

    # A compiled regular expression to efficiently find URLs.
    # This pattern broadly covers URLs starting with 'http(s)://' or 'www.',
    # accounting for common characters found in hostnames, paths, and query parameters.
    _URL_REGEX = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts all identifiable URLs from the provided input data.

        Args:
            data (Any): The input data expected to be a string containing text.
                        If the input is not a string, a warning will be logged,
                        and an empty list will be returned.
            context (Dict[str, Any]): A dictionary containing context-specific information
                                      for the processing node. This node does not
                                      currently utilize any context parameters.

        Returns:
            List[str]: A list of strings, where each string is an extracted URL.
                       Returns an empty list if no URLs are found or if the input
                       data is invalid or empty.
        """
        if not isinstance(data, str):
            logger.warning(
                "[%s] Invalid input type. Expected 'str' but received '%s'. "
                "Returning an empty list of URLs.", self.node_name, type(data).__name__
            )
            return []

        if not data.strip():
            logger.debug("[%s] Input data is an empty or whitespace-only string. Returning an empty list.", self.node_name)
            return []

        try:
            urls = self._URL_REGEX.findall(data)
            if urls:
                logger.info("[%s] Successfully extracted %d URLs from input data.", self.node_name, len(urls))
            else:
                logger.debug("[%s] No URLs found in the provided input data.", self.node_name)
            return urls
        except Exception as e:
            logger.error(
                "[%s] An unexpected error occurred during URL extraction: %s",
                self.node_name, e, exc_info=True
            )
            return []