import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of an input text utterance.

    This node simulates intent classification based on a set of predefined
    keywords and patterns, returning the most likely intent and a confidence score.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode.

        Sets up a simplified mapping of keywords to intents for demonstration.
        In a real-world scenario, this would load a trained machine learning model.
        """
        logger.info("IntentClassifierNode initialized.")
        # Simplified intent patterns for demonstration purposes.
        # In a production environment, this would involve a proper ML model.
        self._intent_patterns = {
            "order_status": ["track my order", "where is my package", "order status", "delivery status"],
            "place_order": ["i want to buy", "order a new", "purchase item", "buy now", "add to cart"],
            "cancel_order": ["cancel my order", "revoke purchase", "undo order", "stop delivery"],
            "greeting": ["hello", "hi there", "good morning", "good evening"],
            "support": ["help me", "support", "technical issue", "contact support"]
        }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Classifies the intent of the input text data.

        The node expects the input `data` to be either a string (the utterance itself)
        or a dictionary containing a 'text' key with the utterance.

        Args:
            data (Any): The input data. Expected to be a string or a dict
                        containing a 'text' key.
            context (Dict[str, Any]): A dictionary containing context-specific data
                                     for the current processing flow. This node
                                     does not currently utilize context but it's
                                     available for future extensions.

        Returns:
            Any: A dictionary containing the original text, the classified intent,
                 and a confidence score. Example:
                 {'text': 'Where is my order?', 'intent': 'order_status', 'confidence': 0.95}

        Raises:
            TypeError: If the input `data` is not a string or a dictionary.
            ValueError: If `data` is a dictionary but does not contain a 'text' key.
        """
        utterance: str = ""

        if isinstance(data, str):
            utterance = data
            logger.debug(f"Received string input: '{utterance}'")
        elif isinstance(data, dict):
            if 'text' not in data:
                logger.error("IntentClassifierNode received a dictionary without a 'text' key.")
                raise ValueError("Input dictionary 'data' must contain a 'text' key for intent classification.")
            utterance = data['text']
            logger.debug(f"Received dictionary input with text: '{utterance}'")
        else:
            logger.error(f"IntentClassifierNode received unsupported data type: {type(data)}")
            raise TypeError("IntentClassifierNode expects input 'data' to be a string or a dictionary with a 'text' key.")

        utterance_lower = utterance.lower()
        predicted_intent: str = "unclassified"
        confidence: float = 0.5  # Default low confidence for unclassified

        # Perform a simple keyword-based intent classification
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern in utterance_lower:
                    predicted_intent = intent
                    # Assign higher confidence if a pattern is directly matched
                    confidence = 0.95
                    logger.debug(f"Identified intent '{intent}' for utterance '{utterance}' based on pattern '{pattern}'.")
                    break
            if predicted_intent != "unclassified":
                break

        if predicted_intent == "unclassified":
            logger.info(f"Could not classify intent for utterance: '{utterance}'. Defaulting to 'unclassified'.")
            # A slightly higher confidence than the default 0.5 to indicate an attempt was made.
            confidence = 0.6

        result = {
            "text": utterance,
            "intent": predicted_intent,
            "confidence": confidence
        }

        logger.info(f"Processed utterance: '{utterance}' -> Intent: '{predicted_intent}', Confidence: {confidence:.2f}")
        return result
