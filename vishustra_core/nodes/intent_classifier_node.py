import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists as per requirement
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.
    This node simulates intent classification based on a set of predefined keyword rules,
    useful for initial routing or simple conversational flows.
    """

    def __init__(self, fallback_intent: str = "GeneralQuery", min_confidence: float = 0.5):
        """
        Initializes the IntentClassifierNode with configurable parameters for its behavior.

        Args:
            fallback_intent: The default intent to return if no specific intent is detected
                             by the keyword rules.
            min_confidence: A threshold for simulated confidence. This value is used as
                            the confidence for the fallback intent.
        """
        self._fallback_intent = fallback_intent
        self._min_confidence = min_confidence
        
        # Define internal intent rules for simulation. In a production system, these
        # would likely be loaded from a configuration file or a trained model.
        self._intent_rules = {
            "BookReservation": ["book", "reserve", "schedule", "plan trip"],
            "CancelReservation": ["cancel", "undo", "revoke", "remove booking"],
            "ModifyReservation": ["change", "update", "modify", "alter booking"],
            "CheckStatus": ["status", "check", "where is my", "is my order"],
            "Greet": ["hello", "hi", "hey", "good morning"],
            "Farewell": ["bye", "goodbye", "see you", "later"],
            "Help": ["help", "support", "assist", "what can you do"],
        }
        logger.info(f"IntentClassifierNode initialized with fallback intent: '{self._fallback_intent}'")


    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        The `data` input is expected to be either a string representing the
        user's query or a dictionary containing a 'text' key with the query string.
        The node performs a simple keyword-based classification, assigning
        a simulated intent and confidence score.

        Args:
            data: The input data, which should contain the text to classify.
                  Expected types: `str` (the query itself) or `Dict[str, Any]`
                  where `data['text']` holds the query string.
            context: A dictionary containing contextual information relevant to the current
                     processing flow. This node uses it for logging and to retrieve
                     a node identifier if available, but does not modify it directly
                     for its output.

        Returns:
            A dictionary containing the classification results:
            - "original_query": The original text input.
            - "intent": The classified intent (e.g., "BookReservation", "GeneralQuery").
            - "confidence": A simulated confidence score for the classification.
                            (0.9 for a direct match, `min_confidence` for fallback).
            - "node_id": The identifier of this node, useful for tracing.

        Raises:
            ValueError: If the input `data` is not in the expected format (str or dict with 'text').
        """
        node_id = context.get("node_id", self.node_name) # Get node ID from context for improved logging

        if isinstance(data, dict) and 'text' in data:
            user_query = str(data['text'])
        elif isinstance(data, str):
            user_query = data
        else:
            logger.error(
                f"[{node_id}] Received invalid data type. "
                f"Expected string or dict with 'text' key. Got: {type(data)} - {data!r}"
            )
            raise ValueError(
                f"Invalid input data for {self.node_name}. "
                f"Expected string or dict with 'text' key."
            )

        processed_query = user_query.lower()
        classified_intent = self._fallback_intent
        confidence = self._min_confidence # Default confidence for fallback

        # Iterate through predefined intent rules to find a match
        for intent, keywords in self._intent_rules.items():
            if any(keyword in processed_query for keyword in keywords):
                classified_intent = intent
                confidence = 0.9 # Higher simulated confidence for a direct keyword match
                break # Found a match, prioritize the first one encountered

        logger.info(
            f"[{node_id}] Query: '{user_query[:75]}{'...' if len(user_query) > 75 else ''}' "
            f"classified as '{classified_intent}' with confidence {confidence:.2f}"
        )

        return {
            "original_query": user_query,
            "intent": classified_intent,
            "confidence": confidence,
            "node_id": node_id, # Include node_id in the output for traceability
        }
