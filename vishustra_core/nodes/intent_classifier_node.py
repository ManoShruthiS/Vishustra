import logging
from typing import Any, Dict, Optional

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node designed to classify the intent of an input text query.

    This node simulates intent classification based on predefined keywords.
    In a production environment, it would integrate with a machine learning
    model (e.g., a transformer-based classifier or a rule-based system)
    to determine the user's underlying goal or request from their input.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the IntentClassifierNode.

        Args:
            config: An optional dictionary containing configuration parameters
                    for the intent classifier, such as model paths, intent definitions,
                    or thresholds. For this simulated node, it can define the
                    keyword-to-intent mappings.
        """
        self._config = config if config is not None else {}
        
        # Load intent mapping from config or use a default
        self._intents_map = self._config.get("intents_map", {
            "book_flight": ["book flight", "reserve ticket", "fly to", "flight booking"],
            "check_status": ["check status", "my order", "where is my", "order status"],
            "cancel_order": ["cancel order", "undo purchase", "revoke subscription"],
            "customer_support": ["help me", "support", "talk to human", "contact agent"],
            "general_inquiry": ["what is", "how to", "information about"],
        })
        self._default_intent = self._config.get("default_intent", "unclassified")
        logger.info(f"IntentClassifierNode initialized. Default intent: '{self._default_intent}'")
        logger.debug(f"Loaded intent map: {self._intents_map}")


    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Expects a string as input data, representing a user query. It simulates
        intent classification by checking for keywords.

        Args:
            data: The input data, expected to be a string (user query).
            context: A dictionary containing contextual information, which might
                     include session data or system configurations.

        Returns:
            A dictionary containing the classified intent and a simulated
            confidence score.
            Example: {"intent": "book_flight", "confidence": 0.95}

        Raises:
            ValueError: If the input `data` is not a string or is empty.
            Exception: For any unforeseen errors during the classification process.
        """
        try:
            if not isinstance(data, str):
                logger.error(
                    f"IntentClassifierNode received invalid data type. Expected 'str', "
                    f"got '{type(data).__name__}'."
                )
                raise ValueError("IntentClassifierNode expects string input data.")

            query = data.strip().lower()

            if not query:
                logger.warning(
                    "Received an empty query string. Assigning default intent "
                    f"'{self._default_intent}' with low confidence."
                )
                return {"intent": self._default_intent, "confidence": 0.3}

            logger.debug(f"Attempting to classify intent for query: '{query}'")

            classified_intent = self._default_intent
            confidence = 0.5  # Base confidence for unclassified or default intent

            # Simulate intent classification by keyword matching
            for intent, keywords in self._intents_map.items():
                if any(keyword.lower() in query for keyword in keywords):
                    classified_intent = intent
                    confidence = 0.9  # Higher confidence for a matched intent
                    break # Assign the first matching intent found

            logger.info(
                f"Query '{query}' classified as intent: '{classified_intent}' "
                f"with confidence: {confidence:.2f}"
            )

            return {"intent": classified_intent, "confidence": confidence}

        except ValueError as ve:
            # Re-raise specific ValueErrors for upstream handling
            raise ve
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred in IntentClassifierNode while processing: {e}"
            )
            # Re-raise other exceptions to ensure pipeline failures are propagated
            raise