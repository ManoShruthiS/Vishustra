import logging
import re
from typing import Any, Dict, List, Set
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from input text data.

    This node can process a single string or an iterable of strings,
    identifying and returning all unique URLs found.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        The input `data` can be:
        - A single string: URLs are extracted directly from this string.
        - An iterable of strings (e.g., list, tuple, set): URLs are extracted
          from each string in the iterable.
        - Any other type: An empty list is returned, and a warning is logged.

        Args:
            data: The input data to process, expected to be a string or iterable of strings.
            context: A dictionary containing contextual information (not used by this node,
                     but required by BaseNode interface).

        Returns:
            A list of unique URLs found in the input data, sorted for consistent output.
        """
        extracted_urls: Set[str] = set()
        
        # Regex pattern for identifying URLs.
        # This pattern is designed to be robust, capturing URLs starting with http(s):// or www.,
        # and handling various common characters found in paths, queries, and fragments,
        # while being mindful of surrounding punctuation by using word boundaries and
        # non-space/non-bracket characters. It's a commonly used pattern for general
        # URL extraction from unstructured text.
        url_pattern = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
        )
        
        try:
            if isinstance(data, str):
                for match in re.finditer(url_pattern, data):
                    extracted_urls.add(match.group(0)) # group(0) is the entire match

            elif isinstance(data, (list, tuple, set)):
                for item in data:
                    if isinstance(item, str):
                        for match in re.finditer(url_pattern, item):
                            extracted_urls.add(match.group(0))
                    else:
                        logger.warning(
                            f"URLExtractorNode encountered non-string item of type "
                            f"{type(item).__name__} within an iterable. Skipping item: {item!r}"
                        )
            else:
                logger.warning(
                    f"URLExtractorNode received unsupported data type: {type(data).__name__}. "
                    "Expected str or iterable of str. Returning empty list."
                )
                return []

            num_urls = len(extracted_urls)
            if num_urls > 0:
                logger.info(f"Successfully extracted {num_urls} unique URLs.")
            else:
                logger.debug("No URLs found in the input data.")

            # Return a sorted list for consistent and predictable output.
            return sorted(list(extracted_urls))

        except Exception as e:
            logger.error(f"An unexpected error occurred during URL extraction: {e}", exc_info=True)
            # In case of an unexpected error, return an empty list to allow the
            # pipeline to continue gracefully, while logging the error for diagnosis.
            return []