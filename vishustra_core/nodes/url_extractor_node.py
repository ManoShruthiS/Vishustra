import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to identify and 
    extract all HTTP and HTTPS URLs from a provided text input using 
    optimized regular expressions.
    """

    # Optimized regex pattern for identifying standard URLs
    _URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for the URL extraction node.
        """
        return "URL_Extractor_Node"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Parses the input data to extract URLs.

        Args:
            data (Any): Expected to be a string containing text to be parsed.
            context (Dict[str, Any]): A dictionary containing orchestration context 
                                      and configuration parameters.

        Returns:
            List[str]: A collection of unique URLs found within the input text.

        Raises:
            ValueError: If the input data is null or empty.
            TypeError: If the input data is not a string.
        """
        try:
            if data is None:
                logger.error("Received null data in URLExtractorNode.")
                raise ValueError("Input data cannot be None.")

            if not isinstance(data, str):
                logger.error(f"Data type mismatch. Expected str, received {type(data).__name__}.")
                raise TypeError(f"URLExtractorNode requires string input, received {type(data).__name__}.")

            logger.debug(f"Starting URL extraction on input of length {len(data)}.")

            # Perform regex findall to retrieve all matches
            extracted_urls = self._URL_PATTERN.findall(data)

            # De-duplicate results while maintaining order if necessary
            unique_urls = list(dict.fromkeys(extracted_urls))

            logger.info(
                f"Extraction complete. Found {len(extracted_urls)} matches "
                f"({len(unique_urls)} unique) in '{self.node_name}'."
            )

            return unique_urls

        except (ValueError, TypeError) as e:
            # Re-raise known validation errors
            raise e
        except Exception as e:
            logger.exception(f"Unexpected failure in {self.node_name}: {str(e)}")
            raise RuntimeError(f"Failed to extract URLs due to an internal error: {str(e)}") from e