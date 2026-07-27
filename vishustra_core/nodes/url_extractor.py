import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node identifies common HTTP/HTTPS URLs within the input string
    and returns them as a list of unique strings.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all unique URLs.

        Args:
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted.
            context: A dictionary containing contextual information relevant
                     to the current processing flow (not directly used by this node).

        Returns:
            A list of unique URLs (strings) found in the data. Returns an empty list
            if the input data is not a string, or if no URLs are found, or in case
            of an internal error.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data of type '%s'. "
                "Expected a string. Returning an empty list of URLs.",
                type(data).__name__
            )
            return []

        # Regex to find http or https URLs. \S+ matches one or more non-whitespace characters.
        # This is a common and reasonably robust pattern for extracting URLs from plain text.
        url_pattern = r'https?://\S+'

        try:
            found_urls = re.findall(url_pattern, data)
            # Use a set to ensure uniqueness, then convert back to a list
            unique_urls = list(set(found_urls))
            
            if unique_urls:
                logger.debug("Successfully extracted %d unique URLs.", len(unique_urls))
            else:
                logger.debug("No URLs found in the provided data.")

            return unique_urls
        except re.error as e:
            logger.error(
                "Regular expression error during URL extraction in URLExtractorNode: %s. "
                "Returning an empty list.", e
            )
            return []
        except Exception as e:
            logger.error(
                "An unexpected error occurred during URL extraction in URLExtractorNode: %s. "
                "Returning an empty list.", e
            )
            return []