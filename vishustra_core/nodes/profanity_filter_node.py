import logging
import re
from typing import Any, Dict, Set, Union

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node designed to filter out profanity from text data.

    This node identifies and replaces known profane words within an input string
    with a specified replacement string, operating in a case-insensitive manner.
    The list of profanities can be customized during initialization or a default
    set will be used.
    """

    def __init__(self, profanity_list: Union[Set[str], None] = None, replacement_string: str = "***"):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list: An optional set of profane words (strings) to filter.
                            If None, a curated default list is used.
                            Words should be provided in lowercase for consistent matching,
                            though the filter itself operates case-insensitively.
            replacement_string: The string to use as a replacement for detected profanities.
                                Defaults to "***".
        """
        self._node_name = "ProfanityFilterNode"
        self._replacement_string = replacement_string
        # Ensure profanity_list is a set for efficient lookup
        self._profanity_list = profanity_list if profanity_list is not None else self._get_default_profanity_list()
        logger.debug(f"{self.node_name} initialized with {len(self._profanity_list)} profanities "
                     f"and replacement string '{self._replacement_string}'.")

    def _get_default_profanity_list(self) -> Set[str]:
        """
        Returns a default set of common profanities (lowercase) for filtering.
        In a production environment, this list would likely be loaded from a
        configuration file, external service, or a dedicated vocabulary management system.
        """
        return {
            "anal", "arse", "ass", "asshole", "bastard", "bitch", "bollocks", "bugger",
            "clit", "cock", "cunt", "damn", "dick", "fag", "fuck", "goddamn", "hell",
            "motherfucker", "nigga", "nigger", "piss", "pussy", "shit", "slut", "son of a bitch",
            "tit", "turd", "wank", "whore"
        }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return self._node_name

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out detected profanities.

        This method expects the 'data' parameter to be a string. If a non-string
        type is provided, a warning is logged, and the original data is returned
        without modification. Profanities are replaced using the configured
        replacement string, respecting word boundaries and ignoring case.

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information relevant to
                     the current processing pipeline. (Currently unused by this node,
                     but available for future extensions like dynamic profanity lists
                     or specific filtering rules).

        Returns:
            The input string with all identified profanities replaced by the
            `replacement_string`, or the original `data` if it was not a string
            or an error occurred during processing.
        """
        if not isinstance(data, str):
            logger.warning(
                f"{self.node_name}: Input data type mismatch. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning original data."
            )
            return data

        processed_text = data
        try:
            for profane_word in self._profanity_list:
                # Construct a regex pattern for whole-word matching, escaping special characters
                # and ensuring case-insensitivity.
                pattern = r'\b' + re.escape(profane_word) + r'\b'
                processed_text = re.sub(pattern, self._replacement_string, processed_text, flags=re.IGNORECASE)

            logger.debug(f"{self.node_name}: Successfully filtered profanities from input.")
            return processed_text
        except Exception as e:
            logger.error(
                f"{self.node_name}: An unexpected error occurred during profanity filtering: {e}",
                exc_info=True  # Logs the full traceback for debugging
            )
            # In case of an error, it's often safer to return the original data
            # rather than failing the entire pipeline.
            return data