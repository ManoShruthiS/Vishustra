import logging
import re
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists in the same conceptual project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from input text data.

    This node expects the input 'data' to be a string. It utilizes a robust
    regular expression to identify and extract common URL patterns (HTTP/HTTPS)
    from the provided text. If the input data is not a string, or if no URLs
    are found, it gracefully returns an empty list.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data string.

        Args:
            data (Any): The input data, expected to be a string containing text
                        from which URLs should be extracted.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing flow. This node
                                       does not directly utilize the context
                                       but adheres to the standard signature.

        Returns:
            List[str]: A list of strings, where each string is an extracted URL.
                       Returns an empty list if no URLs are found, or if the
                       input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data for processing. "
                "Expected a string for URL extraction. Returning an empty list. "
                f"Received data type: {type(data).__name__}."
            )
            return []

        # Regular expression to identify common URL patterns.
        # This pattern captures HTTP and HTTPS schemes, followed by a domain
        # and optional path/query/fragment components, accommodating various
        # valid URL characters including encoded characters.
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        extracted_urls: List[str] = []
        try:
            extracted_urls = url_pattern.findall(data)
            if extracted_urls:
                logger.debug(
                    f"URLExtractorNode successfully extracted {len(extracted_urls)} URLs "
                    f"from the input data."
                )
            else:
                logger.debug("URLExtractorNode found no URLs in the provided data.")
        except re.error as e:
            logger.error(
                f"URLExtractorNode encountered a regular expression error during processing: {e}",
                exc_info=True
            )
            # In case of a regex compilation or matching error, return an empty list
            # to maintain operational stability.
        except Exception as e:
            logger.error(
                f"URLExtractorNode encountered an unexpected error during URL extraction: {e}",
                exc_info=True
            )
            # Catch any other unforeseen exceptions to prevent node failure.

        return extracted_urls
