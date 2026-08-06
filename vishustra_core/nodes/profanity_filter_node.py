import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node responsible for filtering profane language from text data.

    This node identifies a predefined set of profane words and phrases within
    the input string and replaces them with a standardized placeholder (e.g., "****")
    to sanitize content. It handles input validation to ensure only string data
    is processed.
    """

    # A curated, simple list of profanities for demonstration purposes.
    # In a production environment, this list would be externalized (e.g., configuration,
    # database, or a dedicated profanity detection service) and significantly more extensive.
    _PROFANITY_LIST = [
        "ass",
        "bastard",
        "bitch",
        "crap",
        "damn",
        "dick",
        "fuck",
        "hell",
        "shit",
        "piss",
        "wanker",
    ]

    def __init__(self):
        """
        Initializes the ProfanityFilterNode.

        Compiles regular expressions for each profanity to ensure whole-word matching,
        preventing false positives (e.g., "classic" not matching "ass").
        """
        # Compile regex patterns for whole word matching, case-insensitive.
        # \b ensures word boundaries. re.escape handles special characters in profanities.
        self._profanity_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self._PROFANITY_LIST
        ]
        logger.debug(f"[{self.node_name}] Node initialized with {len(self._PROFANITY_LIST)} profanity patterns.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by detecting and replacing known profanities.

        The node iterates through its list of compiled profanity patterns. If a match
        is found in the input `data`, it replaces the detected profanity with "****".

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run. This node does not currently utilize
                     context for configuration but passes it along.

        Returns:
            The processed data (a string with profanities replaced) or raises
            a TypeError if the input is not a string.

        Raises:
            TypeError: If the `data` argument is not an instance of `str`.
        """
        logger.info(f"[{self.node_name}] Initiating profanity filtering process.")

        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Input data must be a string. "
                f"Received type: {type(data).__name__}."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        processed_data = data
        profanity_detected_count = 0

        for pattern in self._profanity_patterns:
            original_state = processed_data
            # Substitute all occurrences of the current profanity pattern
            processed_data = pattern.sub("****", processed_data)

            if original_state != processed_data:
                # If a substitution occurred, increment the detection count.
                # This counts *types* of profanities, not individual instances.
                profanity_detected_count += 1
                logger.debug(
                    f"[{self.node_name}] Filtered pattern '{pattern.pattern}' "
                    f"in the input data."
                )

        if profanity_detected_count > 0:
            logger.info(
                f"[{self.node_name}] Successfully filtered {profanity_detected_count} "
                f"type(s) of profanities from the input."
            )
        else:
            logger.debug(f"[{self.node_name}] No profanities detected in the input data.")

        return processed_data
