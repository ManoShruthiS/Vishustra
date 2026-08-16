import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed to filter out profane language from text data.

    This node identifies and replaces specified profane words with a masked
    sequence (e.g., '***'). It operates on string inputs, performing
    case-insensitive matching and respecting word boundaries to prevent
    unintended replacements within larger words. Non-string inputs are
    logged as warnings and returned unchanged.
    """

    # Default list of profane words. For production environments, this list
    # would typically be loaded from an external configuration source,
    # a dedicated profanity library, or a dynamic database to allow for
    # easy updates and locale-specific adjustments.
    _DEFAULT_PROFANITY_LIST: List[str] = [
        "fuck", "shit", "damn", "asshole", "bitch", "cunt", "motherfucker",
        "piss", "bastard", "dick", "cock", "tits", "bollocks", "wanker"
    ]
    _REPLACEMENT_MASK: str = "***"

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out identified profanities.

        Args:
            data: The input data, which is expected to be a string. If the data
                  is not a string, a warning is logged, and the data is
                  returned without modification.
            context: A dictionary containing contextual information relevant
                     to the processing. This can optionally include a
                     'profanity_list' (List[str]) key, which, if present, will
                     override the node's default profanity list for this
                     specific processing run.

        Returns:
            The processed data with profanities filtered, or the original data
            if it was not a string, an error occurred during filtering, or
            no profanities were found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                "Profanity filtering is only applicable to strings. Returning data unchanged."
            )
            return data

        processed_text = data

        # Retrieve the profanity list to use, prioritizing context-provided list.
        profanity_list_to_use = context.get(
            "profanity_list", self._DEFAULT_PROFANITY_LIST
        )

        logger.debug(
            f"[{self.node_name}] Initiating profanity filtering for text (first 50 chars): "
            f"'{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'"
        )

        try:
            for profane_word in profanity_list_to_use:
                # Construct a regular expression pattern for case-insensitive
                # matching with word boundaries. re.escape() ensures that any
                # special characters within the profane_word itself are treated
                # literally, preventing regex syntax errors.
                pattern = r'\b' + re.escape(profane_word) + r'\b'
                processed_text = re.sub(
                    pattern, self._REPLACEMENT_MASK, processed_text, flags=re.IGNORECASE
                )
            
            logger.info(
                f"[{self.node_name}] Successfully filtered profanities. Result (first 50 chars): "
                f"'{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'"
            )
            return processed_text
        except re.error as e:
            logger.error(
                f"[{self.node_name}] A regular expression error occurred during filtering: {e}. "
                "Returning original data."
            )
            return data
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during profanity filtering: {e}. "
                "Returning original data."
            )
            return data