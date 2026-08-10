import re
import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node designed to extract URLs from textual data.

    This node utilizes a robust regular expression to identify and collect
    HTTP/HTTPS URLs from an input string. It ensures unique URLs are returned
    and provides graceful handling for non-string inputs by attempting
    conversion or logging warnings/errors.
    """

    # A comprehensive regular expression pattern for matching HTTP/HTTPS URLs.
    # This pattern captures common URL structures, including scheme, domain,
    # optional www subdomain, path, query parameters, and fragments.
    _URL_REGEX = (
        r'https?:\/\/(?:www\.)?'  # Scheme (http/https) and optional 'www.'
        r'[-a-zA-Z0-9@:%._\+~#=]{1,256}'  # Domain name
        r'\.[a-zA-Z0-9()]{1,6}'  # Top-level domain
        r'\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'  # Optional path, query, and fragment
    )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "URL Extractor Node"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to identify and return all unique URLs.

        The method expects the input `data` to be a string. If `data` is not
        a string, it attempts to convert it. If conversion fails or an error
        occurs during extraction, appropriate warnings or errors are logged,
        and an empty list is returned.

        Args:
            data: The input payload, typically a string containing text
                  from which URLs need to be extracted. Non-string inputs
                  are robustly handled.
            context: A dictionary providing contextual information for the node's
                     operation. Not directly utilized by this specific node.

        Returns:
            A list of unique strings, where each string represents a URL
            found within the input `data`. Returns an empty list if no URLs
            are found, the input is invalid, or an error occurs.
        """
        logger.debug(f"[{self.node_name}] Starting URL extraction for input data.")

        text_to_process: str
        if not isinstance(data, str):
            try:
                text_to_process = str(data)
                logger.warning(
                    f"[{self.node_name}] Input data is of type '{type(data).__name__}', "
                    f"not a string. Attempting conversion to string for processing."
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Failed to convert input data (type: "
                    f"'{type(data).__name__}') to string: {e}. Returning an empty list."
                )
                return []
        else:
            text_to_process = data

        extracted_urls: List[str] = []
        try:
            # Find all occurrences of the URL pattern in the text
            found_urls = re.findall(self._URL_REGEX, text_to_process)
            # Convert to a set to ensure uniqueness, then back to a list
            extracted_urls = sorted(list(set(found_urls)))
            logger.info(
                f"[{self.node_name}] Successfully extracted {len(extracted_urls)} "
                f"unique URLs from the input."
            )
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during URL extraction: "
                f"{e}. Returning an empty list."
            )

        return extracted_urls