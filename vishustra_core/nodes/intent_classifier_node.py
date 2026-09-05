import logging
from typing import Any, Dict, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that simulates intent classification from a user query.

    This node takes a string as input and attempts to classify its underlying intent
    based on predefined keyword rules. In a production environment, this would
    typically leverage a machine learning model.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode.

        In a real-world scenario, this constructor would load a trained intent
        classification model, configuration, or external resources. For this
        simulation, it sets up a simple dictionary of keyword-to-intent mappings.
        """
        super().__init__()
        # Simulate a small set of intent classification rules based on keywords.
        # This acts as our "model" for demonstration purposes.
        self._intent_rules: List[Tuple[str, str]] = [
            ("order", "place_order"),
            ("buy", "place_order"),
            ("purchase", "place_order"),
            ("status", "check_order_status"),
            ("track", "check_order_status"),
            ("delivery", "check_order_status"),
            ("hello", "greet"),
            ("hi", "greet"),
            ("hey", "greet"),
            ("thank you", "express_gratitude"),
            ("thanks", "express_gratitude"),
            ("support", "contact_support"),
            ("help", "contact_support"),
            ("agent", "contact_support"),
            ("cancel", "cancel_order"),
            ("return", "initiate_return"),
        ]
        logger.info(f"Initialized {self.node_name} with {len(self._intent_rules)} simulated intent rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (user query) to classify its intent.

        The method expects the `data` to be a string representing a user's natural
        language query. It then applies simple keyword matching to determine a
        likely intent and a simulated confidence score.

        Args:
            data: The input data, expected to be a string representing the user's query.
            context: A dictionary containing contextual information for processing.
                     (Currently not used by this simulated node, but available for extensions).

        Returns:
            A dictionary containing the classified 'intent' and a 'confidence' score.
            Example: `{'intent': 'place_order', 'confidence': 0.95}`
            If no specific intent is matched, it defaults to `unknown` with lower confidence.

        Raises:
            ValueError: If the input 'data' is not a string, indicating an incorrect
                        data type for this node's operation.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for {self.node_name}. Expected string for query, "
                f"but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        query = data.lower().strip()
        classified_intent = "unknown"
        confidence = 0.5  # Default confidence for unknown intent or no match

        if not query:
            logger.warning(
                f"Received an empty query in {self.node_name}. Classifying as 'unknown' with minimal confidence."
            )
            return {"intent": "unknown", "confidence": 0.1}

        # Apply simulated keyword-based intent classification
        for keyword, intent in self._intent_rules:
            if keyword in query:
                classified_intent = intent
                confidence = 0.95  # High confidence for a direct keyword match
                break  # Take the first matching rule

        logger.info(
            f"Query '{data}' processed by {self.node_name}, classified as intent "
            f"'{classified_intent}' with confidence {confidence:.2f}."
        )

        return {"intent": classified_intent, "confidence": confidence}
