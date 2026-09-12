
import logging
from typing import Any, Dict

# Assuming vishustra_core is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent
    of a user's query or textual input.

    This node simulates intent classification based on predefined keywords
    and returns a structured result including the classified intent and a
    simulated confidence score.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode with a predefined set of intents
        and associated keywords for simulation purposes.
        In a production environment, this would likely load a trained model
        or configuration from external sources.
        """
        self._intent_patterns = {
            "book_flight": ["book flight", "fly to", "plane ticket", "reserve a flight"],
            "check_weather": ["weather in", "temperature for", "forecast today", "how's the weather"],
            "order_food": ["order pizza", "get food", "delivery", "hunger"],
            "greeting": ["hello", "hi there", "hey", "good morning"],
            "goodbye": ["bye", "see you", "farewell", "good night"],
            "help": ["help me", "support", "assistance"],
            "set_alarm": ["set alarm", "wake me up", "alarm for"],
        }
        logger.debug(f"IntentClassifierNode initialized with patterns: {self._intent_patterns.keys()}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Expected `data` type is a string (user query).
        The `context` dictionary can be used for runtime configuration,
        though not extensively used in this simulated version.

        Args:
            data: The input data, expected to be a string representing a user query.
            context: A dictionary containing contextual information for processing.

        Returns:
            A dictionary with the classified intent and a simulated confidence score,
            e.g., `{"intent": "book_flight", "confidence": 0.9}`.
            Returns `{"intent": "unknown_intent", "confidence": 0.5}` if no intent
            can be confidently classified.

        Raises:
            ValueError: If the input `data` is not a string or is empty.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received non-string data: {type(data)}. Expected string.")
            raise ValueError("Input data for IntentClassifierNode must be a string.")

        if not data.strip():
            logger.warning("IntentClassifierNode received an empty or whitespace-only query.")
            return {"intent": "empty_query", "confidence": 0.0}

        query = data.lower()
        classified_intent = "unknown_intent"
        confidence = 0.5 # Default confidence for unknown intent

        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    classified_intent = intent
                    confidence = 0.95 # Simulate high confidence for a match
                    logger.debug(f"Query '{data}' classified as intent '{classified_intent}' based on pattern '{pattern}'.")
                    return {"intent": classified_intent, "confidence": confidence}

        logger.info(f"Could not classify intent for query: '{data}'. Returning '{classified_intent}'.")
        return {"intent": classified_intent, "confidence": confidence}

