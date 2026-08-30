import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.

    This node simulates intent classification based on a set of predefined keyword rules.
    In a production Vishustra setup, this node would typically integrate with
    an actual NLU service or a fine-tuned machine learning model to provide
    more robust and accurate intent detection.
    """

    # Defines simple keyword-based rules for intent classification.
    # Keys are intent names, values are lists of keywords associated with that intent.
    _intent_rules = {
        "get_weather": ["weather", "forecast", "temperature", "climate"],
        "place_order": ["buy", "order", "purchase", "add to cart", "checkout"],
        "check_status": ["status", "track", "delivery", "where is my"],
        "book_appointment": ["book", "schedule", "appointment", "meeting", "reserve"],
        "customer_support": ["help", "support", "contact", "issue"],
    }
    # Default intent to return if no specific intent is matched by the rules.
    _default_intent = "general_query"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string query) to classify its intent.

        The method iterates through predefined rules to find a matching intent.
        It returns a dictionary containing the classified intent and a simulated confidence score.

        Args:
            data: The input data, expected to be a string representing a user query.
            context: A dictionary containing contextual information relevant to the processing.
                     (Currently not used in this simulation but maintained for interface compatibility).

        Returns:
            A dictionary containing the classified 'intent' (str) and 'confidence' (float).
            Example: {"intent": "get_weather", "confidence": 0.95}

        Raises:
            ValueError: If the input `data` is not a string or is empty after stripping whitespace.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for '{self.node_name}'. Expected string, but received {type(data)}.")
            raise ValueError(f"'{self.node_name}' expects string input, but received {type(data)}.")

        query_stripped = data.strip()
        if not query_stripped:
            logger.warning(f"Received empty string for intent classification in '{self.node_name}'. Returning default intent.")
            return {"intent": self._default_intent, "confidence": 0.0}

        query_lower = query_stripped.lower()
        classified_intent = self._default_intent
        confidence = 0.2  # Default low confidence for fallback intents

        logger.debug(f"Attempting to classify intent for query: '{query_lower}' in node '{self.node_name}'")

        # Simulate intent classification based on keyword presence
        for intent, keywords in self._intent_rules.items():
            for keyword in keywords:
                if keyword in query_lower:
                    classified_intent = intent
                    # Simple heuristic: longer queries matching more specific keywords might have higher confidence
                    # This is purely for simulation purposes.
                    confidence = min(0.9 + (len(keyword) / len(query_lower)) * 0.05, 1.0)
                    logger.debug(f"Matched keyword '{keyword}' for intent '{intent}'.")
                    break  # Found a match for this intent, move to the next intent category
            if classified_intent != self._default_intent:
                break  # A specific intent was found, no need to check further rules

        result = {"intent": classified_intent, "confidence": round(confidence, 2)}
        logger.info(f"Classified intent for query '{data[:75]}{'...' if len(data) > 75 else ''}' as '{result['intent']}' with confidence {result['confidence']}.")
        return result