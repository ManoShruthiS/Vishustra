import logging
import re
from typing import Any, Dict, List

# Simulating the import path for BaseNode as specified in the requirements
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out specified profanities from text data.
    It replaces identified profanities with a customizable string (default: '***').

    This node is designed for text moderation, ensuring output content adheres to
    defined guidelines by sanitizing potentially offensive language.
    """

    def __init__(self, profanity_list: List[str] = None, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode with a list of profanities and a replacement string.

        Args:
            profanity_list (List[str], optional): A list of words to be considered profanity.
                                                  Words are converted to lowercase for matching.
                                                  If None, a default list of common profanities is used.
            replacement_string (str, optional): The string to substitute for detected profanities.
                                                Defaults to '***'.
        """
        self._node_name = "ProfanityFilterNode"
        self._default_profanities = [
            "asshole", "bitch", "bastard", "cunt", "damn", "dick", "fuck", "hell",
            "motherfucker", "shit", "piss", "slut", "whore"
        ]
        # Normalize profanity list to lowercase for case-insensitive matching
        self.profanity_list = [p.lower() for p in (profanity_list or self._default_profanities)]
        self.replacement_string = replacement_string

        logger.debug(
            f"[{self.node_name}] Initialized with {len(self.profanity_list)} "
            f"profanities and replacement string '{self.replacement_string}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return self._node_name

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanities.

        If the input `data` is a string, it iterates through the `profanity_list`
        and replaces any matching words within the string with the `replacement_string`.
        The matching is case-insensitive and respects word boundaries to prevent
        replacing substrings that are not whole words (e.g., 'ass' in 'compass').

        If `data` is not a string, a warning is logged, and the data is returned
        unchanged, as profanity filtering is a text-specific operation.

        Args:
            data (Any): The input data to be processed, typically expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      relevant to the current execution flow.
                                      This node does not directly modify or rely on the context.

        Returns:
            Any: The processed string with profanities filtered, or the original data
                 if it was not a string or if an error occurred during processing.
        """
        logger.info(f"[{self.node_name}] Starting profanity filtering process.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Expected a string for profanity filtering. Returning data as is."
            )
            return data

        processed_text = data
        original_length = len(data)

        for profanity in self.profanity_list:
            # Construct a regex pattern for whole word matching, case-insensitive.
            # re.escape() handles special characters in profanity words (e.g., "f.u.c.k")
            # r'\b' ensures word boundaries, preventing partial word replacements.
            pattern = r'\b' + re.escape(profanity) + r'\b'
            try:
                # Use re.sub to perform case-insensitive replacement across the string.
                processed_text = re.sub(pattern, self.replacement_string, processed_text, flags=re.IGNORECASE)
                logger.debug(f"[{self.node_name}] Replaced occurrences of '{profanity}'.")
            except re.error as e:
                # Log regex compilation/execution errors but continue processing other profanities.
                logger.error(
                    f"[{self.node_name}] Regex error encountered for profanity '{profanity}': {e}. "
                    "This profanity will be skipped."
                )

        logger.info(
            f"[{self.node_name}] Finished profanity filtering. "
            f"Original length: {original_length}, Processed length: {len(processed_text)}."
        )
        return processed_text