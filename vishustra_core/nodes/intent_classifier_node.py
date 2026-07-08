import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node designed to classify the user's intent from input text.

    This node simulates intent classification based on a set of predefined
    keyword rules. It expects the input `data` to be either a string representing
    the user's query or a dictionary containing a 'text' key with the query string.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode with a set of simulated intent rules.
        """
        self._intent_rules = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "order_status": ["order", "status", "track", "delivery", "where is my package"],
            "product_inquiry": ["product", "price", "info", "specifications", "feature", "cost of"],
            "account_management": ["account", "profile", "settings", "password", "username"],
            "support": ["help", "support", "contact us", "technical issue"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"]
        }
        logger.debug(f"IntentClassifierNode initialized with intent rules for: {', '.join(self._intent_rules.keys())}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The node extracts text from the input `data` (either directly as a string
        or from a 'text' key in a dictionary) and attempts to match it against
        predefined keyword rules to determine the user's intent.

        Args:
            data: The input data, expected to be a string (user query) or
                  a dictionary containing a 'text' key with the query string.
            context: A dictionary containing additional contextual information,
                     which can be leveraged for more advanced classification
                     or passed through to subsequent nodes.

        Returns:
            A dictionary containing:
            - 'intent': The classified intent (e.g., "greeting", "order_status", "unknown").
            - 'confidence': A simulated confidence score (e.g., 0.9 for a match, 0.5 for unknown).
            - 'processed_text': The original text that was classified.
            - 'error_message': (Optional) If an error occurred during processing.
            - Any other keys from the original input dictionary (excluding 'text').
        """
        query_text = None
        output_data = {}

        if isinstance(data, str):
            query_text = data
        elif isinstance(data, dict) and 'text' in data and isinstance(data['text'], str):
            query_text = data['text']
            # Carry over other dictionary keys to the output
            output_data.update({k: v for k, v in data.items() if k != 'text'})
        else:
            logger.warning(
                f"IntentClassifierNode received invalid input data type or structure. "
                f"Expected string or dict with 'text' key. Got: {type(data)} -> {data}"
            )
            return {
                "intent": "error",
                "confidence": 1.0,
                "processed_text": str(data),
                "error_message": "Invalid input data format for intent classification.",
                **output_data
            }

        if not query_text.strip():
            logger.info("Received empty query text for intent classification.")
            return {
                "intent": "no_input",
                "confidence": 1.0,
                "processed_text": query_text,
                **output_data
            }

        normalized_query = query_text.lower()
        classified_intent = "unknown"
        confidence = 0.5  # Default confidence for unknown intent

        # Simulate intent classification by checking for keywords
        for intent, keywords in self._intent_rules.items():
            if any(keyword in normalized_query for keyword in keywords):
                classified_intent = intent
                confidence = 0.9  # Higher confidence for a matched intent
                break  # Take the first matched intent for simplicity

        logger.info(
            f"Classified intent for query '{query_text[:75]}{'...' if len(query_text) > 75 else ''}' "
            f"as '{classified_intent}' with confidence {confidence:.2f}"
        )

        return {
            "intent": classified_intent,
            "confidence": confidence,
            "processed_text": query_text,
            **output_data
        }