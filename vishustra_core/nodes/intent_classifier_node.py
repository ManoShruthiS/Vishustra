import logging
from typing import Any, Dict

# Assuming the project structure places base_node in vishustra_core.nodes
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node that simulates intent classification for a given text input.
    It identifies a primary intent based on predefined keyword rules.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode with a set of dummy intent rules.
        In a real-world scenario, this might involve loading an ML model or configuration.
        """
        self._intent_rules: Dict[str, str] = {
            "order": "place_order",
            "status": "check_order_status",
            "return": "initiate_return",
            "cancel": "cancel_order",
            "hello": "greet",
            "hi": "greet",
            "thanks": "thank_you",
            "goodbye": "farewell",
            "help": "get_support",
            "support": "get_support",
            "price": "get_pricing",
            "cost": "get_pricing"
        }
        logger.debug("IntentClassifierNode initialized with dummy intent rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string) to classify its intent.

        Args:
            data (Any): The input data, typically a string representing a user query.
            context (Dict[str, Any]): A dictionary containing context-specific
                                      information for processing.

        Returns:
            Dict[str, Any]: A dictionary containing the original query, the
                            classified intent, and a confidence score. If an
                            error occurs during processing, it includes an 'error'
                            key and sets the intent to 'error'.
        """
        if not isinstance(data, str):
            error_msg = f"Invalid input data type for IntentClassifierNode. Expected string, got {type(data).__name__}."
            logger.error(error_msg)
            return {
                "original_query": data,
                "classified_intent": "error",
                "confidence": 0.0,
                "error": error_msg
            }

        query = data.strip()
        if not query:
            logger.warning("Received an empty query for intent classification.")
            return {
                "original_query": data,
                "classified_intent": "unknown_intent",
                "confidence": 0.1, # Low confidence for empty queries
                "warning": "Empty query received."
            }

        lower_query = query.lower()
        classified_intent = "unknown_intent"
        confidence = 0.5 # Default confidence for unknown or general intent

        for keyword, intent in self._intent_rules.items():
            if keyword in lower_query:
                classified_intent = intent
                confidence = 1.0 # High confidence if a direct keyword match is found
                logger.debug(f"Query '{query[:50]}...' classified as '{intent}' based on keyword '{keyword}'.")
                break # Take the first matching keyword

        if classified_intent == "unknown_intent":
            logger.info(f"Could not definitively classify intent for query: '{query[:50]}...'. Falling back to 'unknown_intent'.")

        # The context dictionary could be used for advanced logic, e.g.,
        # retrieving user-specific preferences or session data for contextual classification.
        # For this simulated node, we merely log its presence if debug is enabled.
        if logger.isEnabledFor(logging.DEBUG) and context:
            logger.debug(f"Context received by IntentClassifierNode: {list(context.keys())}")


        return {
            "original_query": data,
            "classified_intent": classified_intent,
            "confidence": confidence
        }

