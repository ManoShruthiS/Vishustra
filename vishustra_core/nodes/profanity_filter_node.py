import logging
from typing import Any, Dict, List
import re

# This import path is a simulation as per requirements.
# In the actual project structure, this file would reside
# in `vishustra_core/nodes/data_processing/profanity_filter_node.py`
# and BaseNode would be in `vishustra_core/nodes/base_node.py`.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node that filters profanity from text data.

    This node identifies predefined profanity words within the input string
    and replaces them with a masking string (e.g., '***'). It operates
    case-insensitively and uses word boundaries for accurate replacement.
    """

    def __init__(self, replacement_char: str = "***", profanity_list: List[str] = None):
        """
        Initializes the ProfanityFilterNode.

        Args:
            replacement_char (str): The string to replace profanities with.
                                     Defaults to '***'.
            profanity_list (List[str], optional): A custom list of profanity
                                                  words to filter. If None,
                                                  a default list is used.
        """
        self._replacement_char = replacement_char
        # Default, commonly recognized profanities. In a production system,
        # this list would likely be externalized (e.g., configuration, DB, external service).
        self._profanity_list = profanity_list if profanity_list is not None else [
            "shit", "fuck", "damn", "asshole", "bitch", "cunt", "motherfucker", "bastard"
        ]
        
        # Compile regex patterns for efficiency and case-insensitivity.
        # \b ensures whole word matching to avoid censoring parts of innocent words (e.g., "scunthorpe").
        self._profanity_patterns = [
            re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE)
            for p in self._profanity_list
        ]
        logger.debug(f"ProfanityFilterNode initialized with replacement: '{self._replacement_char}' and profanities: {self._profanity_list}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        This method expects the `data` to be a string. It iterates through
        the configured profanity patterns and replaces any matches with the
        specified `replacement_char`.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing context-specific
                                     information for the current processing run.
                                     (e.g., node_id, flow_id). This node uses
                                     'node_id' for improved logging context.

        Returns:
            str: The filtered string with profanities replaced.

        Raises:
            TypeError: If the input `data` is not a string.
            Exception: For any unexpected errors during the filtering process.
        """
        # Extract node_id from context for richer logging, defaulting to node_name
        node_identifier = context.get('node_id', self.node_name)

        if not isinstance(data, str):
            logger.error(
                f"[{node_identifier}] Invalid input type for ProfanityFilterNode. "
                f"Expected string, but received {type(data).__name__}."
            )
            raise TypeError(
                f"ProfanityFilterNode.process expects 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        processed_text = data # Start with the original data
        initial_processed_text = data # Keep a copy for debug logging if needed
        filtered_occurrences = 0

        try:
            for pattern in self._profanity_patterns:
                # Use re.subn to perform replacement and get the count of substitutions.
                new_processed_text, count = re.subn(pattern, self._replacement_char, processed_text)
                if count > 0:
                    filtered_occurrences += count
                    processed_text = new_processed_text

            if filtered_occurrences > 0:
                logger.info(f"[{node_identifier}] Filtered {filtered_occurrences} profanity instances.")
                # Log a snippet of the transformation for debugging
                logger.debug(f"[{node_identifier}] Original (first 100 chars): '{initial_processed_text[:100]}...' -> Filtered (first 100 chars): '{processed_text[:100]}...'")
            else:
                logger.debug(f"[{node_identifier}] No profanities detected in data.")

            return processed_text
        except Exception as e:
            # Log the full traceback for unexpected errors
            logger.exception(
                f"[{node_identifier}] An unexpected error occurred during profanity filtering."
            )
            # Re-raise the exception to allow upstream orchestration to handle it
            raise
