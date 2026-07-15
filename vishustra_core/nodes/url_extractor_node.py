import re
import logging
from typing import Any, Dict, List, Union, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra node that extracts URLs from text data.
    It supports processing single strings or lists of strings and returns a
    unique list of all found URLs.
    """

    # A robust regex for common URL patterns
    # This pattern generally captures URLs starting with http(s):// or www.
    # It attempts to capture a wide range of valid URL characters but stops at whitespace.
    _URL_REGEX = re.compile(
        r"(?i)\b(?:https?://|www\.)[a-z0-9.-]+(?:\.[a-z]{2,})+(?:/[^\s()<>]*|\([\w]+\))*",
        re.IGNORECASE
    )
    # A slightly more permissive regex to catch cases where the URL might end with punctuation
    # but still be part of the URL. This one is often used for general URL extraction.
    _ALTERNATIVE_URL_REGEX = re.compile(r"(https?://\S+)")


    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URL Extractor"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts URLs from the input data.

        Args:
            data: The input data, which can be a single string or a list of strings.
                  Each string is treated as a piece of text from which URLs should be extracted.
            context: A dictionary containing contextual information for the processing.
                     Currently not directly used by this node but available for future extensions.

        Returns:
            A unique list of strings, where each string is a URL found in the input data.
            Returns an empty list if no URLs are found or if the input data is invalid.

        Raises:
            TypeError: If the input 'data' is neither a string nor a list of strings.
        """
        if not isinstance(data, (str, list)):
            logger.error(f"Invalid input data type for URLExtractorNode. Expected str or List[str], got {type(data).__name__}.")
            raise TypeError(
                f"URLExtractorNode expects 'data' to be a string or a list of strings, "
                f"but received type {type(data).__name__}."
            )

        logger.debug(f"Starting URL extraction for data type: {type(data).__name__}")
        all_urls: Set[str] = set()

        text_inputs: List[str]
        if isinstance(data, str):
            text_inputs = [data]
        else:
            text_inputs = data

        for i, text in enumerate(text_inputs):
            if not isinstance(text, str):
                logger.warning(
                    f"Skipping non-string element at index {i} in input list. "
                    f"Expected string, got {type(text).__name__}. "
                    "Only string elements are processed for URL extraction."
                )
                continue

            # Attempt to find URLs using the primary regex
            found_urls = self._URL_REGEX.findall(text)
            if not found_urls:
                # If primary regex doesn't find anything, try the alternative
                found_urls = self._ALTERNATIVE_URL_REGEX.findall(text)

            for url in found_urls:
                # Basic sanitation: remove trailing punctuation that might be wrongly captured
                # if the URL is at the end of a sentence.
                clean_url = url.rstrip('.,;!?')
                all_urls.add(clean_url)
            logger.debug(f"Extracted {len(found_urls)} URLs from text segment {i+1}.")

        extracted_urls_list = sorted(list(all_urls))
        logger.info(f"Finished URL extraction. Found {len(extracted_urls_list)} unique URLs.")
        return extracted_urls_list

# Example of how to use (for local testing/demonstration, not part of the required output)
# if __name__ == "__main__":
#     # This part would typically be handled by the Vishustra orchestration framework
#     # For demonstration purposes:
#     logging.basicConfig(level=logging.INFO)
#     extractor = URLExtractorNode()
#
#     # Test cases
#     test_data_single_string = "Please visit our website at https://www.vishustra.com or check out our docs at docs.vishustra.com/getting-started. Also a github link: https://github.com/vishustra."
#     test_data_list_strings = [
#         "Find more info here: http://example.org/page?id=123. Important!",
#         "Another link: www.google.com, and an incomplete one: ftp://myserver",
#         "No URLs here.",
#         "Also check this one: https://anothersite.net/path/subpath/file.html?query=param&param2=value#section"
#     ]
#     test_data_no_urls = "This is a plain text without any web addresses."
#     test_data_mixed_list = [
#         "Valid URL: https://mixed.com",
#         123, # Non-string element
#         "Another valid: http://mixed2.org"
#     ]
#
#     print(f"\nNode Name: {extractor.node_name}")
#
#     print("\n--- Processing single string ---")
#     urls1 = extractor.process(test_data_single_string, {})
#     print(f"Extracted URLs: {urls1}")
#     assert "https://www.vishustra.com" in urls1
#     assert "docs.vishustra.com/getting-started" in urls1
#     assert "https://github.com/vishustra" in urls1
#
#     print("\n--- Processing list of strings ---")
#     urls2 = extractor.process(test_data_list_strings, {})
#     print(f"Extracted URLs: {urls2}")
#     assert "http://example.org/page?id=123" in urls2
#     assert "www.google.com" in urls2
#     assert "https://anothersite.net/path/subpath/file.html?query=param&param2=value#section" in urls2
#     assert "ftp://myserver" not in urls2 # Current regex might not capture ftp easily unless specified
#
#     print("\n--- Processing text with no URLs ---")
#     urls3 = extractor.process(test_data_no_urls, {})
#     print(f"Extracted URLs: {urls3}")
#     assert len(urls3) == 0
#
#     print("\n--- Processing mixed list with non-string elements ---")
#     urls4 = extractor.process(test_data_mixed_list, {})
#     print(f"Extracted URLs: {urls4}")
#     assert "https://mixed.com" in urls4
#     assert "http://mixed2.org" in urls4
#     assert len(urls4) == 2 # 123 should be skipped
#
#     print("\n--- Testing error handling (invalid data type) ---")
#     try:
#         extractor.process(12345, {})
#     except TypeError as e:
#         print(f"Caught expected error: {e}")
#
#     print("\nAll tests passed (visually and assertions where applicable).")