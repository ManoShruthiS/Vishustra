import re
import logging
from typing import Any, Dict, List

# Importing BaseNode as per Vishustra framework structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from textual data.

    This node identifies common URL patterns, including those starting with
    'http://', 'https://', or 'www.', and compiles them into a list.
    It's robust against common delimiters and non-URL characters.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract all identifiable URLs.

        Args:
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted.
            context: A dictionary containing contextual information relevant
                     to the processing pipeline. This node does not explicitly
                     use the context but adheres to the signature.

        Returns:
            A list of strings, where each string represents a URL found
            within the input `data`. Returns an empty list if no URLs are
            found, or if the input `data` is not a string, or if an
            unexpected error occurs during processing.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type received. Expected 'str', "
                f"but got '{type(data).__name__}'. Returning an empty list of URLs."
            )
            return []

        # A robust regex pattern to capture common URL formats:
        # - `https?://[^\s<>"]+`: Matches URLs starting with 'http://' or 'https://',
        #   followed by any sequence of characters that are not whitespace, '<', '>', or '"'.
        # - `www\.[^\s<>"]+`: Matches URLs starting with 'www.', followed by any sequence
        #   of characters that are not whitespace, '<', '>', or '"'.
        # The `|` operator combines these two patterns.
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'

        extracted_urls: List[str] = []
        try:
            extracted_urls = re.findall(url_pattern, data)
            if extracted_urls:
                logger.info(
                    f"[{self.node_name}] Successfully extracted {len(extracted_urls)} URLs "
                    f"from the input data."
                )
            else:
                logger.debug(
                    f"[{self.node_name}] No URLs found in the input data."
                )
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True  # Logs the full traceback for debugging
            )
            # In case of an unexpected error during regex processing,
            # returning an empty list allows the pipeline to continue gracefully.
            return []

        return extracted_urls