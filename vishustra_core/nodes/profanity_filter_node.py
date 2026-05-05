import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A data processing node for Vishustra designed to filter out offensive language
    from textual data.

    This node identifies a predefined (or user-supplied) list of profanity terms
    within the input text and replaces them with a specified redaction string.
    It intelligently handles case-insensitivity and ensures whole word matching
    to avoid unintended redactions.

    It supports processing plain string data or dictionary objects that contain
    text under a 'text' key.
    """

    def __init__(self, redaction_string: str = "[REDACTED]", profanity_list: Optional[List[str]] = None) -> None:
        """
        Initializes the ProfanityFilterNode with a custom redaction string
        and an optional custom list of profanities.

        Args:
            redaction_string (str): The string used to replace detected profanity.
                                     Defaults to "[REDACTED]".
            profanity_list (Optional[List[str]]): A list of words to identify as profane.
                                                 If None, a default curated list is utilized.
        """
        super().__init__()
        self._redaction_string: str = redaction_string

        if profanity_list is None:
            # A default, commonly recognized list of profanities for demonstration.
            # In a production system, this would typically be loaded from a configuration
            # file, database, or a dedicated profanity library.
            self._profanity_list: List[str] = [
                "fuck", "shit", "bitch", "asshole", "damn", "cunt", "bastard", "pussy", "dick", "wanker"
            ]
        else:
            self._profanity_list = profanity_list

        # Compile a single regex pattern for efficient case-insensitive whole-word matching
        # The `(?:...)` creates a non-capturing group. `re.escape` handles special regex characters.
        # `\b` ensures word boundaries.
        self._profanity_pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(word) for word in self._profanity_list) + r')\b',
            re.IGNORECASE
        )

        logger.debug("ProfanityFilterNode initialized with redaction string '%s' and %d profanity terms.",
                     self._redaction_string, len(self._profanity_list))

    @property
    def node_name(self) -> str:
        """
        Provides the descriptive name for this processing node.
        """
        return "Profanity Filter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact profanity.

        The method expects `data` to be either a string or a dictionary containing
        a 'text' key whose value is a string. Other data types or structures
        will be logged as warnings and returned unmodified.

        Args:
            data (Any): The input data to be filtered.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the processing operation. (Currently unused
                                       by this specific node but required by BaseNode).

        Returns:
            Any: The processed data with profanity replaced by the redaction string,
                 or the original data if no applicable text was found or processed.
        """
        processed_data = data
        original_text: Optional[str] = None
        data_was_dict: bool = False

        if isinstance(data, str):
            original_text = data
            logger.debug("Received string data for profanity filtering.")
        elif isinstance(data, dict) and 'text' in data and isinstance(data['text'], str):
            original_text = data['text']
            # Create a shallow copy to avoid modifying the original input dictionary in place
            processed_data = dict(data)
            data_was_dict = True
            logger.debug("Received dictionary data (key 'text') for profanity filtering.")
        else:
            logger.warning(
                "ProfanityFilterNode received unsupported data type or structure. "
                "Expected 'str' or 'dict' with a 'text' key. Received: '%s'. Returning data unmodified.",
                type(data)
            )
            return data

        if original_text is not None:
            found_profanity = False

            # Define a nested replacer function to update the `found_profanity` flag
            def replacer(match_obj: re.Match) -> str:
                nonlocal found_profanity
                found_profanity = True
                return self._redaction_string

            # Perform the substitution using the compiled regex pattern and the replacer function
            new_cleaned_text = self._profanity_pattern.sub(replacer, original_text)

            if found_profanity:
                # Log the redaction and update the processed_data structure
                if data_was_dict:
                    processed_data['text'] = new_cleaned_text
                    log_msg = (
                        f"Profanity found and redacted in dictionary text. "
                        f"Original start: '{original_text[:50].replace('\n', ' ')}'. "
                        f"Redacted start: '{processed_data['text'][:50].replace('\n', ' ')}'"
                    )
                else:
                    processed_data = new_cleaned_text
                    log_msg = (
                        f"Profanity found and redacted in string. "
                        f"Original start: '{original_text[:50].replace('\n', ' ')}'. "
                        f"Redacted start: '{processed_data[:50].replace('\n', ' ')}'"
                    )
                logger.info(log_msg)
            else:
                logger.debug("No profanity found in the text. Returning original content.")

        return processed_data