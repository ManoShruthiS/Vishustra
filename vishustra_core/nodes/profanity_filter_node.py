import logging
import re
from typing import Any, Dict, List, Union

# Simulate the import path for BaseNode.
# In a fully deployed Vishustra environment, this path would be absolute.
# The try-except block allows for local testing if the core library structure isn't yet in place.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    from abc import ABC, abstractmethod

    class BaseNode(ABC):
        """
        Base class for all Vishustra processing nodes.
        Used as a fallback for local development if vishustra_core is not installed.
        """
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """Processes the input data and returns the result."""
            pass

        @property
        @abstractmethod
        def node_name(self) -> str:
            """Returns the name of the node."""
            pass

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out common profanity from text data.
    It can process single strings or lists of strings, replacing detected profanities
    with a placeholder ('***').
    """

    # A list of profanity patterns to detect. Using raw strings with \b for word boundaries
    # ensures that parts of words (e.g., "shit" in "shitake") are not incorrectly censored.
    _PROFANITY_PATTERNS = [
        r"\b(?:damn|hell|shit|bitch|asshole|fuck|cunt|bastard|pussy|dick)\b"
    ]

    # Compile a single regex pattern for efficiency, using re.IGNORECASE for case-insensitivity.
    # The '|' joins multiple patterns, and the non-capturing group (?:...) makes it cleaner.
    _PROFANITY_REGEX = re.compile(
        "|".join(_PROFANITY_PATTERNS),
        re.IGNORECASE
    )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def _censor_text(self, text: str) -> str:
        """
        Helper method to censor profanity in a single string using the pre-compiled regex.
        """
        return self._PROFANITY_REGEX.sub("***", text)

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        This method supports:
        - `str`: Filters profanity directly in the string.
        - `list[str]`: Iterates through the list, filtering each string element.
          Non-string elements within the list are passed through unchanged, with a warning logged.

        For any other data type, a warning is logged, and the data is returned unchanged.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     (Not directly used by this specific node, but required by BaseNode signature.)

        Returns:
            The processed data with profanity filtered (strings), or the original data
            if its type is not supported for filtering.
        """
        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing single string for profanity filter.")
            return self._censor_text(data)
        elif isinstance(data, list):
            logger.debug(f"[{self.node_name}] Processing list of items for profanity filter.")
            processed_list = []
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    processed_list.append(self._censor_text(item))
                else:
                    logger.warning(
                        f"[{self.node_name}] Item at index {idx} in list is of unsupported type "
                        f"'{type(item).__name__}'. Skipping profanity filter for this item."
                    )
                    processed_list.append(item)  # Keep non-string items as is
            return processed_list
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported data type for profanity filtering: "
                f"'{type(data).__name__}'. Returning data unchanged."
            )
            return data