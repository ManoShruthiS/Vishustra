import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A processing node responsible for identifying and extracting all valid HTTP/HTTPS 
    URLs from a given string input.
    """

    def __init__(self) -> None:
        # Regex pattern for identifying common URL structures
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

    @property
    def node_name(self) -> str:
        """Returns the canonical name of this node."""
        return "url_extractor_node"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Parses the input data to find URLs.
        
        Args:
            data: The input string to parse.
            context: Shared execution context for the pipeline.
            
        Returns:
            A list of unique URLs found in the text.
            
        Raises:
            TypeError: If the input data is not a string.
        """
        logger.debug(f"Node '{self.node_name}' started processing.")

        if not isinstance(data, str):
            error_msg = f"URLExtractorNode expected string input, received {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            urls = self._url_pattern.findall(data)
            
            # Deduplicate while preserving order if necessary
            unique_urls = list(dict.fromkeys(urls))
            
            logger.info(f"Successfully extracted {len(unique_urls)} unique URLs from the input data.")
            return unique_urls

        except Exception as e:
            logger.exception(f"Unexpected error during URL extraction: {str(e)}")
            raise

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.node_name}')>"