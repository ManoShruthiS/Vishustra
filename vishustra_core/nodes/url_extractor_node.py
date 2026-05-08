import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A specialized node for scanning text data and extracting all valid URLs.
    Useful for preprocessing steps in LLM pipelines where external references
    need to be indexed or validated.
    """

    # Robust URL regex pattern (RFC 3986 compliant variant)
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node."""
        return "URLExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Parses the input data for URLs and returns a unique list of findings.

        Args:
            data (Any): The input text to be processed. Expected to be a string.
            context (Dict[str, Any]): The orchestration context, containing 
                                      metadata or global state.

        Returns:
            List[str]: A list of unique URLs found within the text.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "Node %s received invalid data type: %s. Expected: str.", 
                self.node_name, 
                type(data).__name__
            )
            raise TypeError(f"{self.node_name} requires a string input.")

        try:
            logger.debug("Starting URL extraction on input of length %d", len(data))
            
            # Find all matches using the pre-compiled regex
            matches = self.URL_PATTERN.findall(data)
            
            # Deduplicate while preserving order if necessary (set is sufficient for standard discovery)
            unique_urls = list(dict.fromkeys(matches))

            logger.info(
                "Node %s successfully extracted %d unique URLs.", 
                self.node_name, 
                len(unique_urls)
            )
            
            return unique_urls

        except Exception as e:
            logger.exception(
                "Node %s encountered an error during transformation: %s", 
                self.node_name, 
                str(e)
            )
            raise e