
import re
import logging
from typing import Any, Dict, List, Union, Set

# Assuming vishustra_core is available in the project structure
# In a real setup, this would be relative to the project root.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class URLExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts URLs from text input.

    This node can process either a single string or a list of strings,
    identifying and returning all unique URLs found.
    """

    # A comprehensive regex pattern for identifying URLs, including http/https and www prefixes.
    # It aims to capture common URL formats, handling various characters allowed in URLs.
    _URL_PATTERN = re.compile(
        r'(?:https?://|www\.)'  # Scheme (http/https) or www. prefix
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # Domain/Path characters
        r'(?:[/?#]\S*)?' # Optional path, query, fragment
        , re.IGNORECASE
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "URLExtractorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """
        Extracts all unique URLs from the input data.

        The input `data` can be a single string or a list of strings.
        The `context` dictionary is available for potential future extensions
        but is not used for core logic in this version.

        Args:
            data: The input text (string or list of strings) to scan for URLs.
            context: A dictionary containing contextual information for the node.

        Returns:
            A list of unique URLs found in the input data.

        Raises:
            TypeError: If the input `data` is neither a string nor a list of strings.
        """
        logger.debug(f"[{self.node_name}] Processing data for URL extraction. Context: {context.keys()}")

        found_urls: Set[str] = set()

        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Input data is a single string.")
            urls_in_text = self._URL_PATTERN.findall(data)
            found_urls.update(urls_in_text)

        elif isinstance(data, list):
            logger.debug(f"[{self.node_name}] Input data is a list of strings.")
            for i, item in enumerate(data):
                if isinstance(item, str):
                    urls_in_item = self._URL_PATTERN.findall(item)
                    found_urls.update(urls_in_item)
                else:
                    logger.warning(
                        f"[{self.node_name}] List item at index {i} is not a string "
                        f"(type: {type(item).__name__}). Skipping for URL extraction."
                    )
        else:
            error_msg = (
                f"[{self.node_name}] Invalid input data type: expected 'str' or 'List[str]', "
                f"got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"[{self.node_name}] Extracted {len(found_urls)} unique URLs.")
        return sorted(list(found_urls))

