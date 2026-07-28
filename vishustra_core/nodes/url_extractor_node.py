
import re
import logging
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract unique URLs from various text inputs.
    It can process single strings or lists of strings, identifying URLs prefixed
    with 'https://', 'http://', or 'www.'.

    The node normalizes 'www.' prefixed URLs by prepending 'http://' for consistency
    and robustly handles common trailing punctuation.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "URL_Extractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts unique URLs from the input data.

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
                        If a list is provided, non-string elements will be skipped
                        with a debug log message.
            context (Dict[str, Any]): The execution context dictionary. This node
                                      does not currently utilize the context for its
                                      processing logic, but it is available for
                                      future extensions or metadata propagation.

        Returns:
            List[str]: A sorted list of unique URLs found in the data. Returns an empty list
                       if no URLs are found, if the input data is None, or if the input data
                       type is unsupported.
        """
        extracted_urls: Set[str] = set()

        if data is None:
            logger.debug(f"[{self.node_name}] Received None as input data. Returning empty list.")
            return []

        # Robust regex pattern for identifying URLs.
        # It captures URLs starting with 'https://', 'http://', or 'www.'.
        # The pattern handles domain names (including hyphens), TLDs, and common path,
        # query, and fragment characters. It aims to be comprehensive for valid URLs.
        url_pattern = re.compile(
            r"(?P<url>"  # Named group for the entire URL match
            r"(?:https?://|www\.)"  # Match protocol (http/https) or www. prefix
            r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"  # Match domain parts (e.g., example.com)
            r"(?:[a-zA-Z]{2,6}|[a-zA-Z0-9-]{2,})"  # Match Top-Level Domain (e.g., .com, .org)
            r"(?:(?:/[\w\d$&+,/:;=?@#|'~.-]*)*"  # Match optional path components
            r"(?:[?#]\S*)?" # Match optional query string or fragment identifier
            r")?" # Make the path/query/fragment entirely optional for root URLs
            r")"
        )

        def _extract_and_normalize_from_text(text: str) -> Set[str]:
            """
            Helper function to extract and normalize URLs from a single string.
            Applies common cleaning heuristics to improve URL accuracy.
            """
            urls_in_text: Set[str] = set()
            try:
                for match in url_pattern.finditer(text):
                    potential_url = match.group('url')
                    
                    # Basic cleaning: remove common trailing punctuation if it's unlikely
                    # to be part of the URL itself (e.g., "example.com.")
                    # This list of characters covers typical sentence terminators and closing brackets.
                    cleaned_url = potential_url.rstrip(".,;!?'\")>]")

                    # Normalize www. links by prepending 'http://' for consistency in output.
                    if cleaned_url.startswith("www.") and not cleaned_url.startswith(("http://", "https://")):
                        cleaned_url = f"http://{cleaned_url}"
                    
                    # Add the cleaned and normalized URL only if it's not empty.
                    if cleaned_url:
                        urls_in_text.add(cleaned_url)

            except Exception as e:
                # Log the error with full traceback for debugging, but continue processing
                # if possible for other parts of the input.
                logger.warning(
                    f"[{self.node_name}] Error during URL extraction from text "
                    f"(first 100 chars: '{text[:100]}...'). Error: {e}", exc_info=True
                )
            return urls_in_text

        if isinstance(data, str):
            extracted_urls.update(_extract_and_normalize_from_text(data))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    extracted_urls.update(_extract_and_normalize_from_text(item))
                else:
                    logger.debug(
                        f"[{self.node_name}] Skipping non-string item at index {i} "
                        f"(type: {type(item).__name__}) in input list for URL extraction."
                    )
        else:
            # Log a warning for unsupported input types and return an empty list.
            logger.warning(
                f"[{self.node_name}] Unsupported input data type for {self.node_name}: {type(data).__name__}. "
                "Expected str or list[str]. Returning empty list."
            )
            return []

        # Convert the set of unique URLs to a sorted list for a deterministic output order.
        return sorted(list(extracted_urls))

