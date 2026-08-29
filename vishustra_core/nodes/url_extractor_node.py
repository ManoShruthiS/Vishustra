import re
import logging
from typing import Any, Dict, List

# Assuming BaseNode is available from this path as per Vishustra's project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node responsible for extracting URLs from input text data.

    This node uses a regular expression to identify common URL patterns, including
    those prefixed with HTTP, HTTPS, FTP, or 'www.'. It's designed to be robust
    for URLs embedded within free-form text.
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
            data: The input data, expected to be a string containing text
                  from which URLs should be extracted.
            context: A dictionary providing contextual information for the processing.
                     This node does not currently utilize the context, but it's
                     required by the BaseNode interface.

        Returns:
            A list of strings, where each string is a unique URL found in the input data.
            Returns an empty list if no URLs are found, or if the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning an empty list."
            )
            return []

        # Regex pattern for URLs. This pattern is designed to capture common web (http/https),
        # FTP, and 'www.'-prefixed URLs. It accounts for domain structures, TLDs (2-63 chars),
        # and optional paths, queries, or fragments.
        # This regex balances robustness for general text with avoiding overly aggressive matches.
        url_pattern = re.compile(
            r'\b'  # Word boundary to prevent partial matches within words
            r'(?:https?://|ftp://|www\.)'  # Protocols (http, https, ftp) or www. prefix
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # One or more domain/subdomain labels
            r'[a-zA-Z]{2,63}'  # Top-level domain (TLD) - 2 to 63 characters long
            r'(?:/?|[/?]\S+)'  # Optional trailing slash or path/query/fragment (non-whitespace characters)
            r'\b'  # Another word boundary
        )

        try:
            extracted_urls = url_pattern.findall(data)

            if extracted_urls:
                # Using a set to ensure unique URLs and then converting back to a list
                unique_urls = list(set(extracted_urls))
                logger.debug(
                    f"[{self.node_name}] Successfully extracted {len(unique_urls)} "
                    f"unique URL(s) from the input data."
                )
                return unique_urls
            else:
                logger.debug(f"[{self.node_name}] No URLs found in the provided input data.")
                return []
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: {e}. "
                "Returning an empty list to maintain pipeline stability."
            )
            return []