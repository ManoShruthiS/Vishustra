import logging
import re
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A processing node for Vishustra that filters out profanity from text data.

    This node replaces detected offensive words with a string of asterisks (***).
    It supports a default list of profanities and allows augmentation or override
    of this list via the processing context for dynamic control.
    """

    _default_profanity_list: List[str]

    def __init__(self, initial_profanity_list: Optional[List[str]] = None):
        """
        Initializes the ProfanityFilterNode with a base list of profanities.

        Args:
            initial_profanity_list (Optional[List[str]]): An optional list of
                additional profanities to include with the node's defaults.
        """
        super().__init__()
        # Initialize a core set of profanities. In a production system, this
        # might be loaded from a configuration file or a dedicated service.
        self._default_profanity_list = [
            "fuck", "shit", "asshole", "bitch", "cunt", "damn", "bastard", "pussy",
            "motherfucker", "fucker", "cock", "dick", "slut", "whore", "ass", "crap"
        ]

        if initial_profanity_list:
            # Add unique items from the provided initial list, ensuring lower case
            self._default_profanity_list.extend([p.lower() for p in initial_profanity_list])

        # Remove duplicates and sort by length in descending order.
        # This helps in preventing shorter words (e.g., "ass") from being matched
        # before longer words (e.g., "asshole"), ensuring the more specific
        # and often longer offensive terms are replaced first.
        self._default_profanity_list = sorted(
            list(set(self._default_profanity_list)),
            key=len,
            reverse=True
        )
        logger.debug(
            f"{self.node_name} initialized with {len(self._default_profanity_list)} "
            "default profanities."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profanity.

        The method expects string input. It identifies and replaces
        profane words with '***'. The profanity list can be augmented
        or overridden through the `context` dictionary by providing
        a 'profanity_list' key.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing. Can include 'profanity_list'
                                      to provide custom profanities for this run.

        Returns:
            Any: The processed data with profanities filtered, or the original
                 data if it's not a string and an error is raised.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"{self.node_name} received non-string data: {type(data)}. Expected string."
            )
            raise TypeError(
                f"{self.node_name} requires string input, but received {type(data).__name__}"
            )

        original_data = data
        processed_data = data

        # Get profanity list from context, or use the node's default list.
        # This allows for dynamic profanity list updates per processing call.
        context_profanity_list_raw = context.get("profanity_list", [])

        # Combine default and context-provided lists
        current_profanity_list = list(self._default_profanity_list)
        if context_profanity_list_raw:
            current_profanity_list.extend([p.lower() for p in context_profanity_list_raw])
            # Re-sort and remove duplicates for the combined list
            current_profanity_list = sorted(
                list(set(current_profanity_list)),
                key=len,
                reverse=True
            )

        found_profanities = []
        for word in current_profanity_list:
            # Use regex for whole-word, case-insensitive matching.
            # '\b' ensures word boundaries, preventing partial matches like 'ass' in 'classic'.
            # re.escape() handles special characters in the profanity word itself.
            pattern = r'\b' + re.escape(word) + r'\b'

            # Search for the pattern to identify *which* words were filtered
            if re.search(pattern, processed_data, re.IGNORECASE):
                found_profanities.append(word)
                # Replace found profanity with asterisks.
                # flags=re.IGNORECASE makes the replacement case-insensitive.
                processed_data = re.sub(pattern, "***", processed_data, flags=re.IGNORECASE)

        if found_profanities:
            # Log unique profanities found to avoid redundant logging if a word appears multiple times.
            unique_found = sorted(list(set(found_profanities)))
            logger.info(
                f"{self.node_name} filtered profanities: {', '.join(unique_found)} from input."
            )
            logger.debug(f"Original: '{original_data}' -> Processed: '{processed_data}'")
        else:
            logger.debug(f"{self.node_name} found no profanities in the input data.")

        return processed_data