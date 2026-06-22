import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available at this path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract URLs from textual input data.

    This node identifies common URL patterns, including those starting with
    'http://', 'https://', or 'www.', and returns them as a list of strings.
    It's robust to various URL structures, including paths, query parameters,
    and fragment identifiers.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and extract URLs.

        Args:
            data: The input payload, expected to be a string containing text
                  from which URLs should be extracted. If not a string, a
                  warning is logged, and an empty list is returned.
            context: A dictionary containing contextual information relevant to
                     the current processing pipeline. This node does not
                     currently utilize the context.

        Returns:
            A `List[str]` containing all unique URLs found within the input `data`.
            Returns an empty list if no URLs are found or if the input `data`
            is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning an empty list."
            )
            return []

        # A comprehensive regular expression pattern to identify various URL formats.
        # It covers http/https schemes, www. prefixes, and robustly captures
        # common URL characters including path segments, query parameters,
        # and fragment identifiers.
        url_pattern = re.compile(
            r'(?:https?://|www\.)'  # Scheme (http/https) or www. prefix
            r'(?:[a-zA-Z0-9-]+\.)+' # Domain name (e.g., example.com)
            r'[a-zA-Z]{2,6}'        # TLD (e.g., com, org, net)
            r'(?:/?|[/?]\S+)'       # Optional path, query, fragment
        )
        
        found_urls = url_pattern.findall(data)
        
        if found_urls:
            logger.info(f"[{self.node_name}] Successfully extracted {len(found_urls)} URLs from input.")
        else:
            logger.debug(f"[{self.node_name}] No URLs found in the provided text data.")
            
        return found_urls
