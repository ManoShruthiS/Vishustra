import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A specialized node designed to parse input text and extract all valid URLs 
    using regular expression matching. This is useful for pre-processing 
    steps in web-crawling or information retrieval pipelines.
    """

    # Comprehensive URL regex pattern to match standard protocols and domains
    _URL_REGEX = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "url_extractor_node"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Scans the input data for URLs.
        
        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): The current pipeline execution context.
            
        Returns:
            List[str]: A list of unique URLs discovered in the text. Returns an empty 
                      list if input is invalid or no matches are found.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string input of type %s. Skipping extraction.", 
                type(data).__name__
            )
            return []

        try:
            logger.debug("Starting URL extraction on input payload.")
            
            # Find all matches based on the regex
            matches = self._URL_REGEX.findall(data)
            
            # Use dict keys to remove duplicates while preserving original discovery order
            unique_urls = list(dict.fromkeys(matches))
            
            logger.info(
                "Successfully processed URL extraction. Found %d unique URLs.", 
                len(unique_urls)
            )
            return unique_urls

        except Exception as e:
            logger.error("Failed to extract URLs due to an unexpected error: %s", str(e), exc_info=True)
            # Return an empty list to maintain pipeline stability
            return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.node_name}')>"

```python
# Example usage within the Vishustra framework:
# node = URLExtractorNode()
# urls = node.process("Check out https://github.com and http://python.org", {})
