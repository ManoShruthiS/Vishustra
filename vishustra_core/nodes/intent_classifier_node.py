import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.
    This node simulates intent classification based on simple keyword matching.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input data based on predefined rules.

        Args:
            data: The input data, expected to be a string representing a user query
                  or a piece of text requiring intent classification.
            context: A dictionary containing contextual information for processing.
                     While this specific node implementation doesn't directly use
                     the context for classification, it is passed through as part
                     of the node's output to maintain framework consistency.

        Returns:
            A dictionary containing:
            - "input": The original input data.
            - "classified_intent": A string representing the identified intent.
            - "context": The original context dictionary.

            Example: {"input": "What's the weather like?", "classified_intent": "utility.weather", "context": {...}}

        Raises:
            TypeError: If the input 'data' is not a string, which is required for text classification.
            ValueError: If an unexpected error occurs during the classification process.
        """
        classified_intent: str = "general.unknown"

        logger.debug(f"[{self.node_name}] Starting intent classification for input type: {type(data).__name__}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected 'str', but received '{type(data).__name__}'.")
            raise TypeError(
                f"IntentClassifierNode expects string input for classification, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.info(f"[{self.node_name}] Received empty or whitespace-only string for classification. Assigning 'general.unknown'.")
            return {"input": data, "classified_intent": classified_intent, "context": context}

        try:
            lower_data = data.lower().strip()

            # A simple, extensible mapping of keywords to intents.
            # In a real-world scenario, this would be backed by a sophisticated ML model.
            intent_keywords = {
                "ecommerce.purchase": ["buy", "order", "purchase", "shop for", "add to cart"],
                "customer_service.support": ["help", "support", "issue", "problem", "assist me", "troubleshoot"],
                "general.greeting": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
                "utility.weather": ["weather", "forecast", "temperature", "climate", "is it raining"],
                "information.query": ["what is", "how to", "tell me about", "who is", "where is", "explain"],
                "booking.reservation": ["book", "reserve", "appointment", "schedule", "make a booking"],
                "navigation.direction": ["directions to", "how do I get to", "route for"],
            }

            for intent, keywords in intent_keywords.items():
                for keyword in keywords:
                    if keyword in lower_data:
                        classified_intent = intent
                        break
                if classified_intent != "general.unknown":
                    break
            
            # Log a truncated version of the input for clarity and to prevent excessively long log lines
            log_input = data if len(data) <= 100 else f"{data[:97]}..."
            logger.info(f"[{self.node_name}] Classified intent for input '{log_input}' as '{classified_intent}'.")
            
            return {"input": data, "classified_intent": classified_intent, "context": context}

        except Exception as e:
            # Catch any unexpected errors during the classification logic
            logger.exception(f"[{self.node_name}] An unexpected error occurred during intent classification for input: '{data}'.")
            raise ValueError(f"Error classifying intent: {e}") from e