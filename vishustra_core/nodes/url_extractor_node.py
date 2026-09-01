import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from textual data.

    This node leverages regular expressions to identify common URL patterns
    within the input string, providing a list of all detected URLs.
    It gracefully handles non-string inputs and cases where no URLs are found.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "URL Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Args:
            data: The input data, expected to be a string (e.g., document content,
                  message body). If not a string, a warning is logged, and
                  an empty list is returned.
            context: A dictionary containing contextual information relevant
                     to the orchestration. This node does not directly use
                     the context for its core logic but adheres to the signature.

        Returns:
            A list of strings, where each string is a URL identified within
            the input `data`. Returns an empty list if no URLs are found
            or if the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning an empty list."
            )
            return []

        # A robust regular expression to capture URLs with common schemes (http, https, ftp)
        # or starting with 'www.', followed by non-whitespace characters.
        # The word boundaries (\b) help prevent partial matches.
        url_pattern = r'\b(?:https?://|ftp://|www\.)\S+\b'

        try:
            extracted_urls = re.findall(url_pattern, data)

            if not extracted_urls:
                logger.debug(f"[{self.node_name}] No URLs found in the provided data.")
            else:
                logger.info(
                    f"[{self.node_name}] Successfully extracted {len(extracted_urls)} URL(s)."
                )
            return extracted_urls
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}",
                exc_info=True  # Logs the full traceback for debugging
            )
            return []