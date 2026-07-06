
import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent of a user query.

    This node simulates intent classification based on predefined keywords. In a production
    environment, this would typically involve integration with an NLP model.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode with a predefined mapping of keywords to intents.
        """
        self._intent_keyword_map: Dict[str, str] = {
            "book flight": "book_flight",
            "reserve a flight": "book_flight",
            "check status": "check_flight_status",
            "flight status": "check_flight_status",
            "ticket status": "check_ticket_status",
            "hello": "greet",
            "hi there": "greet",
            "goodbye": "farewell",
            "thank you": "gratitude",
            "cancel booking": "cancel_booking",
            "modify booking": "modify_booking",
            "change reservation": "modify_booking",
            "help me": "request_help",
            "support": "request_help"
        }
        logger.debug(f"[{self.node_name}] Initialized with {len(self._intent_keyword_map)} intent mappings.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Classifies the intent of the input text data.

        This method expects a string as input data, representing a user query.
        It iterates through internal keyword mappings to find a matching intent.
        If no specific intent is found, it defaults to "unknown".

        Args:
            data: The input data, expected to be a string representing the user's query.
            context: A dictionary containing contextual information for the processing.
                     (Currently not used for classification logic but available for future use,
                     e.g., dynamic intent maps or user-specific settings).

        Returns:
            A string representing the classified intent (e.g., "book_flight", "unknown").

        Raises:
            TypeError: If the input data is not a string.
        """
        logger.info(f"[{self.node_name}] Starting intent classification for input data type: {type(data)}")

        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        text_query = data.lower().strip()
        classified_intent = "unknown"

        # Simulate intent classification using simple keyword matching
        # This approach prioritizes earlier matches in the map
        for keyword, intent in self._intent_keyword_map.items():
            if keyword in text_query:
                classified_intent = intent
                logger.debug(
                    f"[{self.node_name}] Matched keyword '{keyword}' for intent '{intent}' "
                    f"in query fragment: '{text_query[:100]}...'"
                )
                break  # Found a match, assume this is the primary intent

        logger.info(
            f"[{self.node_name}] Classified intent: '{classified_intent}' "
            f"for query fragment: '{text_query[:100]}...'"
        )
        return classified_intent
