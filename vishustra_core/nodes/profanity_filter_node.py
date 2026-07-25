import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out specified profane words from text data.
    It replaces identified profane words with a configurable placeholder string.

    The list of profane words and the replacement string can be provided via the
    context dictionary, allowing for flexible and dynamic configuration.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Filters profane words from the input text data, replacing them with a
        placeholder. The filtering is case-insensitive and uses whole-word matching.

        Configuration can be provided through the `context` dictionary:
        - `profanity_list` (Optional[list[str]]): A list of words to identify as profane.
                                                  If not provided, a default list is used.
        - `replacement_string` (Optional[str]): The string to substitute for each
                                                profane word found. Defaults to '***'.

        Args:
            data: The input data, expected to be a string, to be filtered.
            context: A dictionary containing configuration parameters for the node.

        Returns:
            The processed string with profane words replaced.

        Raises:
            TypeError: If the input `data` is not a string, as this node specifically
                       operates on text.
        """
        if not isinstance(data, str):
            logger.error(
                f"ProfanityFilterNode received invalid input type. "
                f"Expected 'str', but got '{type(data).__name__}'. "
                f"This node is designed for text processing."
            )
            raise TypeError(
                f"ProfanityFilterNode requires string input, "
                f"but received {type(data).__name__}."
            )

        # Retrieve profanity list from context or use a robust default
        default_profanity_list = ["fuck", "shit", "damn", "hell", "bitch", "cunt", "asshole"]
        profanity_list = context.get("profanity_list", default_profanity_list)
        if not isinstance(profanity_list, list) or not all(isinstance(w, str) for w in profanity_list):
            logger.warning(
                "Provided 'profanity_list' in context is invalid or malformed. "
                "Falling back to default profanity list."
            )
            profanity_list = default_profanity_list

        # Retrieve replacement string from context or use a default
        replacement_string = context.get("replacement_string", "***")
        if not isinstance(replacement_string, str):
            logger.warning(
                "Provided 'replacement_string' in context is not a string. "
                "Falling back to default replacement string '***'."
            )
            replacement_string = "***"


        filtered_text = data
        logger.debug(f"Starting profanity filtering for text of length {len(data)}.")

        for word in profanity_list:
            # Construct a regex pattern for whole-word, case-insensitive matching.
            # re.escape() handles any special regex characters within the profane word.
            # \\b ensures word boundaries, preventing partial word replacement (e.g., 'shit' in 'shitake').
            pattern = r"\b" + re.escape(word) + r"\b"
            
            # Perform the replacement.
            original_text_before_replacement = filtered_text
            filtered_text = re.sub(pattern, replacement_string, filtered_text, flags=re.IGNORECASE)

            if original_text_before_replacement != filtered_text:
                logger.debug(f"Replaced instances of '{word}' with '{replacement_string}'.")

        logger.info("Profanity filtering process completed.")
        return filtered_text