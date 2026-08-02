
import re
import logging
from typing import Any, Dict, List

# Assuming BaseNode is available at this path as defined in the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node uses regular expressions to identify and extract common URL patterns
    (e.g., those starting with http(s):// or www.) from an input string.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data: The input data, expected to be a string containing text.
                  If `data` is not a string, a warning will be logged, and an
                  empty list will be returned.
            context: A dictionary of contextual information. This node does not
                     currently utilize the context dictionary, but it is provided
                     as part of the `BaseNode` interface.

        Returns:
            A list of strings, where each string is a URL found in the input data.
            Returns an empty list if no URLs are found or if the input data is
            not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data (type: %s). Expected a string for URL extraction. "
                "Returning an empty list.", type(data).__name__
            )
            return []

        # Regular expression to match URLs.
        # It broadly covers 'http(s)://' or 'www.' followed by non-whitespace characters.
        # This pattern aims for a balance between comprehensiveness and avoiding
        # overly complex or potentially mis-matching patterns for general text.
        url_pattern = re.compile(
            r'(?:https?://|www\.)'  # Matches 'http://', 'https://', or 'www.'
            r'\S+'                  # Matches one or more non-whitespace characters for the rest of the URL
        )

        try:
            extracted_urls = url_pattern.findall(data)
            if not extracted_urls:
                logger.debug("No URLs found in the provided text data.")
            else:
                logger.debug("Successfully extracted %d URLs.", len(extracted_urls))
            return extracted_urls
        except re.error as e:
            # This specific error indicates an issue with the regex pattern itself,
            # which should ideally be caught during development/testing.
            logger.error("Regular expression error encountered in URLExtractorNode: %s", e)
            return []
        except Exception as e:
            # Catch any other unexpected errors during the findall operation.
            logger.error(
                "An unexpected error occurred during URL extraction in URLExtractorNode: %s",
                e, exc_info=True
            )
            return []

