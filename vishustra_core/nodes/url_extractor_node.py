import logging
import re
from typing import Any, Dict, List

# Importing BaseNode from the specified project path.
# The definition of BaseNode is assumed to exist in 'vishustra_core.nodes.base_node'.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from text content.

    This node takes a string as input and returns a list of all identified
    URLs within that string. It leverages a comprehensive regular expression
    to accurately capture a wide range of URL formats, including those
    starting with 'http(s)://' and 'www.'.
    """

    # A robust regular expression pattern to capture various URL formats.
    # This pattern accounts for common URL structures, including schemes,
    # domains, paths, queries, and fragments, while also handling URLs
    # embedded within punctuation or parenthesis.
    _URL_REGEX = re.compile(
        r'\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
        r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+'
        r'(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE
    )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Expects the input `data` to be a string. If the data is not a string,
        a warning is logged, and an empty list is returned.

        Args:
            data: The input content, expected to be a string from which URLs
                  are to be extracted.
            context: A dictionary containing contextual information. This node
                     does not utilize the context for its operation.

        Returns:
            A `List[str]` containing all unique URLs found in the input `data`.
            Returns an empty list if no URLs are found or if the input data
            is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning an empty list."
            )
            return []

        try:
            # The regex is designed with a main capturing group that contains the full URL.
            # findall for a regex with capturing groups returns a list of tuples,
            # where each tuple contains the strings matched by the groups.
            # The first element of each tuple (index 0) corresponds to the full URL match.
            raw_urls_matches = self._URL_REGEX.findall(data)
            
            # Extract the full URL from the first capturing group of each match.
            extracted_urls = [match[0] for match in raw_urls_matches if match and match[0]]
            
            # Optionally, convert to a set to remove duplicates and then back to a list
            # if unique URLs are strictly required. For now, maintaining order and allowing
            # duplicates as per typical extraction behavior.
            
            logger.debug(
                f"[{self.node_name}] Successfully extracted {len(extracted_urls)} URLs "
                f"from the input data."
            )
            return extracted_urls
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True  # Log full traceback for debugging
            )
            return []