import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node designed to filter out specified profane words from input text.

    This node replaces identified profanities with a configurable replacement string,
    ensuring text content adheres to desired moderation standards. It performs
    case-insensitive, whole-word matching to avoid unintended replacements.
    """

    def __init__(self, profanity_list: Optional[List[str]] = None, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode with a list of words to filter and
        a string to replace them with.

        Args:
            profanity_list: An optional list of strings considered profane. If None,
                            a default, illustrative list is used. Words are stored
                            and matched case-insensitively.
            replacement_string: The string used to replace detected profanities.
                                Defaults to '***'.
        """
        super().__init__()
        self._default_profanity_list = [
            "badword", "damn", "asshole", "fuck", "shit", "bitch", "cunt", "motherfucker", "bastard"
        ]
        
        # Ensure profanity list is processed to lowercase for consistent matching
        self._profanity_list = [
            word.lower() for word in (profanity_list if profanity_list is not None else self._default_profanity_list)
        ]
        self._replacement_string = replacement_string
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._profanity_list)} profanities "
            f"and replacement string: '{self._replacement_string}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, identifying and filtering out profane words.

        The method expects the input `data` to be a string. It iterates through
        the configured profanity list, performing case-insensitive, whole-word
        replacement.

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information, which can be
                     used for logging or future extensions but is not directly
                     modified by this node.

        Returns:
            The processed string with all identified profanities replaced by
            the configured `replacement_string`.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'. Context: {context}"
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, "
                f"but received {type(data).__name__}."
            )

        processed_data = data
        found_profanity = False

        for profanity in self._profanity_list:
            # Construct a regex pattern for whole-word matching, case-insensitive
            # re.escape() handles special regex characters in profanity words
            pattern = r'\b' + re.escape(profanity) + r'\b'
            
            # Use re.sub to replace all occurrences globally (g) and case-insensitively (i)
            new_data = re.sub(pattern, self._replacement_string, processed_data, flags=re.IGNORECASE)
            
            if new_data != processed_data:
                found_profanity = True
                processed_data = new_data
        
        if found_profanity:
            logger.info(
                f"[{self.node_name}] Profanity detected and filtered in data. "
                f"Original data hash for reference: {hash(data)}."
            )
        else:
            logger.debug(
                f"[{self.node_name}] No profanity detected in data. "
                f"Data hash for reference: {hash(data)}."
            )

        return processed_data