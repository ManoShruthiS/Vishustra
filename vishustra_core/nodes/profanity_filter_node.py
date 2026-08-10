import logging
import re
from typing import Any, Dict

# BaseNode is expected to be available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node designed to filter profanity from string data.
    It identifies specified profane words (case-insensitively, whole word match)
    and replaces them with '***'.
    """

    # A simple, illustrative list of profane words.
    # In a production-grade system, this list would typically be:
    # 1. Externalized (e.g., loaded from a configuration file or a database).
    # 2. Configurable (allowing users to add/remove words).
    # 3. More comprehensive, potentially including variations or allowing
    #    for different levels of strictness.
    _PROFANE_WORDS = [
        "fuck", "shit", "bitch", "asshole", "damn", "cunt",
        "piss", "bastard", "motherfucker", "cock", "dick", "wanker"
    ]

    def __init__(self):
        """
        Initializes the ProfanityFilterNode.
        Pre-compiles regex patterns for each profane word to optimize
        performance during the `process` method, ensuring efficient and
        case-insensitive whole-word replacement.
        """
        self._compiled_patterns = []
        for word in self._PROFANE_WORDS:
            # We use '\b' for word boundaries to ensure whole word matching.
            # re.escape() is used to handle any special regex characters that might
            # be present in the profane words themselves, preventing regex errors.
            # re.IGNORECASE makes the pattern match regardless of letter casing.
            self._compiled_patterns.append(re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE))
        logger.debug(f"[{self.node_name}] Initialized with {len(self._PROFANE_WORDS)} words for filtering.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Profanity Filter Node"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profane words.

        If the input `data` is a string, it iterates through its internal list
        of profane words and replaces any matches (case-insensitive, whole word)
        with '***'. For non-string data, a warning is logged, and the original
        data is returned without modification. Robust error handling is included
        to manage unexpected issues during processing.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        for filtering to occur.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the current workflow execution. This can
                                       include session IDs, user preferences, etc.

        Returns:
            Any: The processed data (a string with profanity replaced by '***'),
                 or the original `data` if it was not a string or if an error
                 occurred during the filtering process.
        """
        logger.info(f"[{self.node_name}] Starting profanity filtering process for incoming data.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is of type '{type(data).__name__}', not a string. "
                "Profanity filtering is applicable only to strings. Returning original data."
            )
            return data

        processed_data = data
        try:
            # Iterate through all compiled regex patterns and apply them sequentially
            for pattern in self._compiled_patterns:
                # The .sub() method replaces all non-overlapping occurrences of the pattern
                # with the replacement string ('***').
                processed_data = pattern.sub('***', processed_data)
            
            if processed_data != data:
                logger.debug(
                    f"[{self.node_name}] Profanity detected and filtered. "
                    f"Original data length: {len(data)}, Filtered data length: {len(processed_data)}."
                )
            else:
                logger.debug(f"[{self.node_name}] No profane words detected in the input data.")

            logger.info(f"[{self.node_name}] Profanity filtering completed successfully.")
            return processed_data

        except Exception as e:
            # Catching a broad exception to ensure robustness against unforeseen issues.
            # In a more granular implementation, specific exceptions might be caught.
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during profanity filtering: {e}",
                exc_info=True  # This includes traceback in the log for detailed debugging.
            )
            # In error scenarios, it's generally safer to return the original, unprocessed
            # data to avoid introducing corrupted or partially processed data into the
            # subsequent nodes in the orchestration pipeline.
            return data