import logging
import re
from typing import Any, Dict, List, Union

# Assuming this import path based on the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from input text data.

    It can process a single string or a list of strings, identifying
    common URL patterns (http(s)://... and www....) and returning
    a unique list of discovered URLs.
    """

    # Regex pattern to identify URLs. This pattern is designed to be robust
    # for extraction, covering common HTTP/HTTPS schemes, optional 'www.',
    # domain names, TLDs, and a wide range of common path, query, and
    # fragment characters, stopping at whitespace or end of input.
    _URL_PATTERN = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&//=]*)'
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        The input `data` can be a single string or a list of strings.
        The `context` dictionary is currently not used but is required by the BaseNode interface.

        Args:
            data: The input data, which can be a string containing text
                  or a list of strings.
            context: A dictionary containing contextual information for processing.
                     Not directly used by this node but passed through.

        Returns:
            A list of unique URLs found in the input data. Returns an empty list
            if no URLs are found or if the input data is invalid/empty.

        Raises:
            TypeError: If the input `data` is neither a string nor a list of strings,
                       or if a list contains non-string elements.
        """
        extracted_urls: List[str] = []
        
        if not data:
            logger.debug(f"[{self.node_name}] Received empty or None data. Returning an empty list.")
            return []

        text_inputs: List[str]
        if isinstance(data, str):
            text_inputs = [data]
        elif isinstance(data, list):
            # Ensure all elements in the list are strings for consistent processing
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. "
                    f"Expected List[str], but found mixed types."
                )
                raise TypeError(
                    f"[{self.node_name}] Input list must contain only strings. "
                    f"Found non-string elements in provided data."
                )
            text_inputs = data
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data).__name__}. "
                f"Expected str or List[str]."
            )
            raise TypeError(
                f"[{self.node_name}] Invalid input data type. Expected str or List[str], "
                f"got {type(data).__name__}."
            )

        for text_segment in text_inputs:
            try:
                found_in_segment = self._URL_PATTERN.findall(text_segment)
                if found_in_segment:
                    extracted_urls.extend(found_in_segment)
            except Exception as e:
                # Log any unexpected errors during regex processing of a segment
                logger.warning(
                    f"[{self.node_name}] Error during URL extraction for a text segment. "
                    f"Error: {e}. Segment starts with: '{text_segment[:100]}...'"
                )

        # Remove duplicates while preserving the order of first appearance
        unique_urls = list(dict.fromkeys(extracted_urls))

        if not unique_urls:
            logger.debug(f"[{self.node_name}] No URLs found in the provided data after processing.")
        else:
            logger.info(f"[{self.node_name}] Successfully extracted {len(unique_urls)} unique URLs.")
            logger.debug(f"[{self.node_name}] Extracted URLs: {unique_urls}")

        return unique_urls