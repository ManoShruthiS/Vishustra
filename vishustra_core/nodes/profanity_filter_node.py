from vishustra_core.nodes.base_node import BaseNode
import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters profanity from text data.
    It replaces specified profane words with a configurable placeholder string.
    This node operates on string inputs and ensures robustness against invalid data types.
    """

    # A default set of profanities. This can be extended or replaced via configuration.
    DEFAULT_PROFANITIES: Set[str] = {
        "badword", "cursing", "damn", "ass", "bitch", "fuck", "shit", "bastard",
        "hell", "crap", "piss", "motherfucker"
    }
    # The default string used to replace detected profanities.
    REPLACEMENT_STRING: str = "****"

    def __init__(self, custom_profanities: List[str] = None, replacement_string: str = None):
        """
        Initializes the ProfanityFilterNode with optional custom profanities
        and a custom replacement string.

        Args:
            custom_profanities: An optional list of strings representing profane words.
                                If provided, these words will be used instead of the
                                default set. Words will be converted to lowercase internally.
            replacement_string: An optional string to use as the placeholder for
                                detected profanities. Defaults to "****" if None.
        
        Raises:
            TypeError: If `custom_profanities` is not a list or `replacement_string` is not a string.
        """
        if custom_profanities is not None:
            if not isinstance(custom_profanities, list):
                logger.error(
                    "Initialization error for ProfanityFilterNode: 'custom_profanities' "
                    f"must be a list of strings, but received {type(custom_profanities).__name__}."
                )
                raise TypeError("Expected 'custom_profanities' to be a list of strings.")
            # Convert to a set for efficient lookup and store as lowercase
            self._profanities: Set[str] = {word.lower() for word in custom_profanities if isinstance(word, str)}
            if len(self._profanities) != len(custom_profanities):
                logger.warning("Some non-string items were found and ignored in custom_profanities list.")
            logger.debug(f"ProfanityFilterNode initialized with custom profanities: {sorted(list(self._profanities))}")
        else:
            self._profanities = self.DEFAULT_PROFANITIES
            logger.debug("ProfanityFilterNode initialized with default profanities.")

        self._replacement_string: str = replacement_string if replacement_string is not None else self.REPLACEMENT_STRING
        if not isinstance(self._replacement_string, str):
            logger.error(
                "Initialization error for ProfanityFilterNode: 'replacement_string' "
                f"must be a string, but received {type(self._replacement_string).__name__}."
            )
            raise TypeError("Expected 'replacement_string' to be a string.")
        
        # Pre-compile regex patterns for each profanity for efficiency and case-insensitivity
        # using \b for word boundaries to avoid partial matches (e.g., 'ass' in 'passage')
        self._profanity_patterns: List[re.Pattern] = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self._profanities
        ]
        logger.debug(f"ProfanityFilterNode using replacement string: '{self._replacement_string}'")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out profanity.

        The method expects `data` to be a string. It iterates through the
        configured profanity list and replaces each instance (case-insensitive)
        with the specified replacement string.

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information for the node.
                     This node does not directly utilize context for its filtering logic,
                     but it's available for logging or future extensions.

        Returns:
            A string with all detected profanities replaced by the replacement string.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"ProfanityFilterNode received invalid input type. "
                f"Expected 'str', but got '{type(data).__name__}'. Data: {data!r}"
            )
            raise TypeError(f"ProfanityFilterNode can only process string data, received {type(data).__name__}.")

        sanitized_data = data
        for pattern in self._profanity_patterns:
            sanitized_data = pattern.sub(self._replacement_string, sanitized_data)
        
        if sanitized_data != data:
            logger.info(
                f"ProfanityFilterNode successfully filtered text. "
                f"Original length: {len(data)}, Sanitized length: {len(sanitized_data)}."
            )
        else:
            logger.debug("ProfanityFilterNode processed text, no profanity found or filtered.")

        if context:
            logger.debug(f"ProfanityFilterNode received context with keys: {list(context.keys())}.")

        return sanitized_data