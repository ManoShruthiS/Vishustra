import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available from this path in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from a given string.

    It identifies common URL patterns including HTTP(S) and 'www.' prefixed links.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor Node"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Expected 'data' type is a string containing the text to be scanned.
        If 'data' is not a string, a warning is logged, and an empty list is returned.

        Args:
            data: The input data, expected to be a string.
            context: A dictionary containing contextual information for processing.
                     Currently not used by this node but available for future extensions.

        Returns:
            A list of strings, where each string is a URL found in the input data.
            Returns an empty list if no URLs are found or if the input data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data of type '%s'. "
                "Expected a string for URL extraction. Returning an empty list.",
                type(data).__name__
            )
            return []

        # Regular expression to match URLs.
        # This regex looks for:
        # - http:// or https:// followed by any non-whitespace characters
        # - www. followed by any non-whitespace characters that contain at least one dot
        # It's reasonably robust for common web URLs.
        url_pattern = re.compile(
            r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        try:
            extracted_urls = url_pattern.findall(data)
            if not extracted_urls:
                logger.debug("No URLs found in the provided data string.")
            else:
                logger.debug("Successfully extracted %d URLs.", len(extracted_urls))
            return extracted_urls
        except re.error as e:
            logger.error(
                "Regular expression error during URL extraction in URLExtractorNode: %s", e
            )
            return []
        except Exception as e:
            logger.error(
                "An unexpected error occurred during URL extraction in URLExtractorNode: %s", e
            )
            return []

```
**Self-Correction/Refinement during thought process:**

*   **Initial Regex thought:** `r'https?://(?:www\.)?\S+\.\S+'` - This is too simplistic. It might miss URLs without `www.` or with complex paths, query parameters, etc.
*   **Refined Regex thought:** `r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'` - This is much better. It handles a broader set of URL characters including common special characters and URL-encoded parts. I also decided to combine `http/https` and `www.` separately using an `OR` (`|`) to catch both patterns explicitly.
*   **Data Type Check:** Initially, I might just apply regex. But the `BaseNode` requires robust error handling. Explicitly checking `isinstance(data, str)` and logging a warning/returning an empty list is crucial for robust backend components.
*   **Context Usage:** The prompt specifies `context: Dict[str, Any]`. While this node doesn't *use* it, it's good practice to mention it in the docstring and explain its current non-usage, hinting at future extensibility (e.g., passing custom regex patterns via context).
*   **Logging Levels:** Use `logger.warning` for bad input, `logger.error` for exceptions, and `logger.debug` for successful but less critical operations (like "X URLs found" or "no URLs found"). This helps manage log verbosity.
*   **Return Type:** Explicitly set the return type of `process` to `List[str]`.
*   **Docstrings:** Ensure comprehensive docstrings for the class and the `process` method, explaining its purpose, arguments, and return values.
*   **Exception Handling:** Added a `try-except` block for `re.error` specifically, and a general `Exception` catch for any unforeseen issues during the regex operation. This makes the node more resilient.
*   **Import Path:** Re-verified the import path for `BaseNode` as per the prompt's `vishustra_core.nodes.base_node`.# COMMIT_MSG: feat: Add URLExtractorNode for robust URL extraction
# FILENAME: vishustra_core/nodes/url_extractor_node.py
import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available from this path in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from a given string.

    It identifies common URL patterns including HTTP(S) and 'www.' prefixed links.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "URL Extractor Node"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract URLs.

        Expected 'data' type is a string containing the text to be scanned.
        If 'data' is not a string, a warning is logged, and an empty list is returned.

        Args:
            data: The input data, expected to be a string.
            context: A dictionary containing contextual information for processing.
                     Currently not used by this node but available for future extensions.

        Returns:
            A list of strings, where each string is a URL found in the input data.
            Returns an empty list if no URLs are found or if the input data is not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                "URLExtractorNode received non-string data of type '%s'. "
                "Expected a string for URL extraction. Returning an empty list.",
                type(data).__name__
            )
            return []

        # Regular expression to match URLs.
        # This regex looks for:
        # - URLs starting with http:// or https://, followed by any valid URL characters.
        # - URLs starting with www., followed by any valid URL characters.
        # It's designed to be reasonably robust for common web URLs.
        url_pattern = re.compile(
            r'\b(?:https?://|www\.)'  # Start with http(s):// or www.
            r'(?:[a-zA-Z0-9-]+\.)+'   # Domain name parts (e.g., example.com)
            r'(?:[a-zA-Z]{2,6})'      # Top-level domain (e.g., com, org, net)
            r'(?:[/][^\s]*)?'         # Optional path, query, fragment (anything non-whitespace)
            r'\b'                     # Word boundary to prevent partial matches
        )


        try:
            extracted_urls = url_pattern.findall(data)
            if not extracted_urls:
                logger.debug("No URLs found in the provided data string.")
            else:
                logger.debug("Successfully extracted %d URLs.", len(extracted_urls))
            return extracted_urls
        except re.error as e:
            logger.error(
                "Regular expression error during URL extraction in URLExtractorNode: %s", e
            )
            return []
        except Exception as e:
            logger.error(
                "An unexpected error occurred during URL extraction in URLExtractorNode: %s", e
            )
            return []