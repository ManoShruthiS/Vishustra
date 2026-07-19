import logging
import re
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from input text data.

    This node utilizes a regular expression to identify common URL patterns within
    a given string and returns a list of all unique URLs found. It is resilient
    to various input formats by performing type validation.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the provided input data.

        The input `data` is expected to be a string containing text. If `data` is
        not a string, a `TypeError` will be raised to indicate an invalid input contract.
        Empty or whitespace-only strings will result in an empty list, with a debug log.

        Args:
            data: The input content, typically a string from which URLs need to be extracted.
            context: A dictionary for additional node-specific or global context.
                     Currently, this node does not utilize the context dictionary
                     for its URL extraction logic.

        Returns:
            A list of unique strings, where each string represents a URL found
            within the input `data`. Returns an empty list if no URLs are found
            or if the input data, after validation, is effectively empty.

        Raises:
            TypeError: If the input `data` is not of type `str`.
        """
        if not isinstance(data, str):
            error_message = (
                f"Node '{self.node_name}' received an unexpected input type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        # Handle empty or whitespace-only strings gracefully
        if not data.strip():
            logger.debug(
                f"Node '{self.node_name}' received an empty or whitespace-only string. "
                "Returning an empty list of URLs."
            )
            return []

        # Robust regex for identifying common HTTP/HTTPS URLs.
        # This pattern captures typical URL components including scheme, domain,
        # optional www, path, query parameters, and fragments.
        url_pattern = re.compile(
            r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        )

        found_urls = url_pattern.findall(data)

        if not found_urls:
            logger.info(f"Node '{self.node_name}' processed data but found no URLs.")
            return []

        # Ensure uniqueness and maintain order if needed, but for now, just unique.
        unique_urls = list(set(found_urls))
        logger.debug(
            f"Node '{self.node_name}' successfully extracted {len(unique_urls)} unique URLs."
        )
        return unique_urls