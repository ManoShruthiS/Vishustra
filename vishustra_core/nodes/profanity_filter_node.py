import logging
import re
from typing import Any, Dict, List, Optional

# Assuming vishustra_core.nodes.base_node is available in the Python path
# The BaseNode definition from the project context provides its interface.
from vishustra_core.nodes.base_node import BaseNode


class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out profanity from text data.

    This node identifies predefined profane words within an input string and replaces
    them with a configurable placeholder. It performs case-insensitive, whole-word
    matching to ensure accurate and context-aware filtering.
    """

    def __init__(
        self,
        profanity_list: Optional[List[str]] = None,
        replacement: str = "***"
    ):
        """
        Initializes the ProfanityFilterNode with a specific list of profanities
        and a replacement string.

        Args:
            profanity_list (Optional[List[str]]): A list of words to be considered profane.
                                                  If None, a sensible default list is used.
                                                  Words should be lowercase for internal consistency,
                                                  but matching is case-insensitive.
            replacement (str): The string used to substitute detected profanities.
                               Defaults to '***'.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Use a default list if none is provided, ensuring it's comprehensive but not exhaustive.
        self.profanity_list: List[str] = profanity_list if profanity_list is not None else [
            "fuck", "shit", "asshole", "bitch", "cunt", "damn", "hell", "piss", "bastard"
        ]
        self.replacement = replacement
        
        self.logger.debug(
            f"ProfanityFilterNode initialized with profanities: {self.profanity_list} "
            f"and replacement: '{self.replacement}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by filtering out profanity.

        If the input `data` is a string, the method identifies and replaces all
        occurrences of words from the configured profanity list with the
        specified replacement string. Matching is case-insensitive and aims for
        whole-word boundaries. Non-string inputs are logged and returned untouched.

        Args:
            data (Any): The input data to be processed. Expected to be a string for filtering.
            context (Dict[str, Any]): A dictionary containing contextual information relevant
                                       to the current processing flow (e.g., user ID, session data).

        Returns:
            Any: The processed data with profanity filtered, or the original data
                 if it was not a string or an error occurred during filtering.
        """
        if not isinstance(data, str):
            self.logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering skipped."
            )
            return data

        filtered_data = data
        profanity_detected = False

        try:
            for profanity in self.profanity_list:
                # Construct a regex pattern for whole-word, case-insensitive matching.
                # re.escape() handles any special regex characters in the profanity word.
                # \b ensures that only whole words are matched (e.g., 'ass' won't match 'assistant').
                pattern = r'\b' + re.escape(profanity) + r'\b'
                
                # Check if the profanity exists before attempting replacement.
                # This helps in logging efficiency and clarity.
                if re.search(pattern, filtered_data, re.IGNORECASE):
                    self.logger.info(
                        f"[{self.node_name}] Profanity '{profanity}' detected. Applying filter."
                    )
                    # Perform the replacement, ensuring case-insensitivity.
                    filtered_data = re.sub(pattern, self.replacement, filtered_data, flags=re.IGNORECASE)
                    profanity_detected = True
            
            if profanity_detected:
                self.logger.info(
                    f"[{self.node_name}] Profanity filtering completed. "
                    f"Original length: {len(data)}, Filtered length: {len(filtered_data)}."
                )
            else:
                self.logger.debug(f"[{self.node_name}] No profanity detected in data.")

        except Exception as e:
            self.logger.error(
                f"[{self.node_name}] An unexpected error occurred during profanity filtering: {e}",
                exc_info=True  # Log full traceback for debugging
            )
            # In case of an unhandled error, return the original data to prevent pipeline failure.
            return data

        return filtered_data