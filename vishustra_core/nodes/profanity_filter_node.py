import logging
import re
from typing import Any, Dict, List

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that sanitizes text data by filtering out
    specified profanity words and replacing them with a placeholder.
    This node is designed for content moderation, compliance, and data cleaning workflows.
    It performs case-insensitive, whole-word matching to reduce false positives.
    """

    def __init__(self, profanity_list: List[str] = None, replacement_char: str = "*"):
        """
        Initializes the ProfanityFilterNode with a configurable list of profanity words
        and a character for replacement.

        Args:
            profanity_list (List[str], optional): A list of words to filter.
                                                  Each word in the list should be in lowercase,
                                                  as matching is performed case-insensitively.
                                                  If None, a default list of common profanities is used.
            replacement_char (str): The character to use for replacing each letter of a
                                    filtered profanity word. For example, if '*' is used,
                                    "fuck" would become "****". Defaults to '*'.
        """
        self._profanity_list = [word.lower() for word in profanity_list] if profanity_list is not None else self._get_default_profanity_list()
        
        if not self._profanity_list:
            logger.warning(
                f"[{self.node_name}] Initialized without any profanity words. "
                "The node will not perform any filtering operations."
            )
        self._replacement_char = replacement_char
        logger.debug(f"[{self.node_name}] Initialized with {len(self._profanity_list)} profanity words for filtering.")

    def _get_default_profanity_list(self) -> List[str]:
        """
        Provides a sensible default list of common profanity words.
        In a production environment, this list would typically be loaded from
        a dynamic source like a configuration service, a database, or an external
        moderation API for easier updates and maintenance.
        """
        return [
            "ass", "bastard", "bitch", "bollocks", "bugger", "bullshit", "chink",
            "cock", "cunt", "damn", "dick", "douche", "dyke", "fuck", "goddamn",
            "hell", "jerk", "motherfucker", "nigga", "nigger", "piss", "prick",
            "pussy", "shit", "slut", "son of a bitch", "tits", "twat", "wank", "whore"
        ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, replacing identified profanity words with a
        placeholder string. The filtering is case-insensitive and utilizes
        regular expressions with word boundaries to ensure whole-word matching,
        thereby reducing unintended replacements of substrings within legitimate words.

        Args:
            data (Any): The input data, primarily expected to be a string for effective filtering.
                        If the input is not a string, a warning will be logged, and the
                        original data will be returned unmodified.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This node currently
                                       does not utilize the context for its core filtering logic.

        Returns:
            Any: The processed data (a string with profanity filtered out) or the
                 original data if it was not a string or if no profanity was found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering skipped; returning original data without modification."
            )
            return data

        processed_text = data
        total_filtered_instances = 0

        for word in self._profanity_list:
            # Construct a regex pattern for whole-word matching.
            # re.escape is crucial to treat any special regex characters in 'word' as literals.
            # \b ensures that only whole words are matched (e.g., 'hell' matches "go to hell" but not "hello").
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Create the replacement string, e.g., "****" for "fuck"
            replacement_str = self._replacement_char * len(word)
            
            # Perform the replacement using re.subn, which returns the new string and
            # the number of substitutions made. re.IGNORECASE ensures case-insensitive matching.
            new_text, num_replacements = re.subn(
                pattern,
                replacement_str,
                processed_text,
                flags=re.IGNORECASE
            )
            processed_text = new_text
            total_filtered_instances += num_replacements

        if total_filtered_instances > 0:
            logger.info(
                f"[{self.node_name}] Filtered {total_filtered_instances} instances of profanity "
                f"in text. (Original length: {len(data)}, New length: {len(processed_text)})."
            )
        else:
            logger.debug(f"[{self.node_name}] No profanity found in the text input.")

        return processed_text
