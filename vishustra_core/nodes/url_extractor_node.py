import re
import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from textual input data.

    This node uses a regular expression to identify and collect unique HTTP,
    HTTPS, and 'www.' prefixed URLs present in the input string.
    It handles cases where the input is not a string gracefully.
    """

    # Compiled regular expression pattern for robust URL extraction.
    # It covers:
    # - 'http://' or 'https://' scheme
    # - 'www.' prefix
    # - Domain name characters (alphanumeric, hyphen, dot)
    # - Path, query, fragment characters (various common symbols)
    # - Negative lookbehind `(?<![.,?!])` to prevent capturing trailing punctuation
    #   if the URL ends a sentence or is followed by common delimiters.
    _URL_REGEX = re.compile(
        r'(?:https?://|www\.)'  # Match http://, https://, or www.
        r'[a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;%]+' # Match common URL characters
        r'(?<![.,?!])' # Negative lookbehind to avoid trailing punctuation like '.', '!', '?'
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        The `data` input is expected to be a string. If it's not a string,
        a warning is logged, and an empty list is returned.
        The `context` dictionary is currently not utilized by this node but
        is part of the BaseNode interface.

        Args:
            data: The input data, typically a string containing text from which
                  URLs need to be extracted.
            context: A dictionary holding contextual information relevant to the
                     current processing pipeline.

        Returns:
            A list of unique strings, each representing a URL found in the input data.
            Returns an empty list if no URLs are found, or if the input data
            is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"URLExtractorNode received non-string data of type '{type(data).__name__}'. "
                "Expected a string for URL extraction. Returning an empty list."
            )
            return []

        try:
            # Find all non-overlapping matches of the URL pattern in the data.
            # Convert to a set to ensure uniqueness, then back to a list.
            found_urls = list(set(self._URL_REGEX.findall(data)))
            logger.debug(f"Successfully extracted {len(found_urls)} unique URLs.")
            return found_urls
        except Exception as e:
            # Catch any unexpected errors during regex processing
            logger.error(
                f"An unexpected error occurred during URL extraction: {e}",
                exc_info=True # Log traceback for detailed debugging
            )
            return []

