import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode is located in vishustra_core.nodes.base_node as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text data.

    This node identifies and extracts fully qualified URLs (http/https/ftp/ftps/mailto/news)
    and common 'www.' prefixed URLs from input strings or lists of strings.
    It uses a robust regular expression to cover a wide range of URL formats
    while attempting to avoid common trailing punctuation.
    """

    # A robust regular expression for common URL patterns.
    # It covers:
    # 1. URLs with standard schemes (http, https, ftp, ftps, mailto, news).
    # 2. URLs starting with 'www.'.
    # It accounts for domain names, TLDs (up to 63 characters for modern TLDs),
    # and optional paths, queries, and fragments.
    # A negative lookbehind `(?<![.,?!;])` is included to prevent capturing
    # trailing punctuation commonly found at the end of sentences.
    URL_REGEX = re.compile(
        r'(?:'  # Start non-capturing group for OR logic
            r'(?:https?|ftps?|mailto|news):' # Scheme-based URLs
            r'[-a-zA-Z0-9@:%._\+~#=]{1,256}'  # Subdomain/domain characters
            r'\.[a-zA-Z0-9()]{1,63}'          # TLD (1 to 63 chars)
            r'\b'                             # Word boundary after TLD
            r'(?:'                            # Optional path/query/fragment part
                r'[-a-zA-Z0-9()@:%_\+.~#?&//=]*'
                r'(?<![.,?!;])'               # Exclude trailing punctuation
            r')?'
        r')'
        r'|'  # OR
        r'(?:'  # Start non-capturing group for www. URLs
            r'www\.'                          # www. prefix
            r'[-a-zA-Z0-9@:%._\+~#=]{1,256}'  # Subdomain/domain characters
            r'\.[a-zA-Z0-9()]{1,63}'          # TLD
            r'\b'                             # Word boundary after TLD
            r'(?:'                            # Optional path/query/fragment part
                r'[-a-zA-Z0-9()@:%_\+.~#?&//=]*'
                r'(?<![.,?!;])'               # Exclude trailing punctuation
            r')?'
        r')'
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Args:
            data (Any): The input data. Expected to be a string or a list of strings.
                        Non-string items within a list will be skipped with a warning.
                        Other top-level data types will result in an empty list and a warning.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. This node does not directly
                                       use the context for URL extraction, but it's
                                       passed through as per the BaseNode API.

        Returns:
            List[str]: A list of unique URLs found in the input data.
                       Returns an empty list if no URLs are found or
                       if the input data type is not supported for extraction.
        """
        extracted_urls: List[str] = []

        if isinstance(data, str):
            urls_in_text = self.URL_REGEX.findall(data)
            extracted_urls.extend(urls_in_text)
            if urls_in_text:
                logger.debug(f"[{self.node_name}] Extracted {len(urls_in_text)} URLs from string input.")
            else:
                logger.debug(f"[{self.node_name}] No URLs found in string input.")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    urls_in_item = self.URL_REGEX.findall(item)
                    extracted_urls.extend(urls_in_item)
                    if urls_in_item:
                        logger.debug(f"[{self.node_name}] Extracted {len(urls_in_item)} URLs from list item at index {i}.")
                    else:
                        logger.debug(f"[{self.node_name}] No URLs found in list item at index {i}.")
                else:
                    logger.warning(
                        f"[{self.node_name}] Skipping non-string item at index {i} "
                        f"in list input (type: {type(item).__name__}). Only strings are processed."
                    )
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported data type for URL extraction: "
                f"{type(data).__name__}. Expected str or List[str]. Returning empty list."
            )
            return [] # Return an empty list for unsupported types

        # Convert to a set to ensure uniqueness, then back to a list.
        # Sorting for consistent output, though not strictly required by the contract.
        unique_urls = sorted(list(set(extracted_urls)))
        
        logger.info(f"[{self.node_name}] Successfully extracted {len(unique_urls)} unique URLs.")
        return unique_urls