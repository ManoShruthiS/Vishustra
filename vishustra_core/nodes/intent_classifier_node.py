import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a given text input.
    
    This node simulates intent classification using a keyword-based approach for demonstration.
    In a production environment, this would typically integrate with an actual
    intent classification model (e.g., a fine-tuned LLM or a dedicated NLU service)
    to provide more robust and nuanced intent recognition.
    """

    _DEFAULT_INTENT_MAPPING: Dict[str, List[str]] = {
        "book_flight": ["book flight", "reserve flight", "ticket", "travel to", "departure", "arrival", "flights"],
        "check_status": ["status of", "where is my", "tracking number", "shipment", "order status", "delivery"],
        "customer_support": ["help", "support", "contact us", "problem", "issue", "assist me", "trouble"],
        "account_management": ["my account", "login", "password reset", "profile", "settings", "change email"],
        "general_inquiry": ["what is", "how to", "information about", "explain", "tell me about"]
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text based on keyword matching.

        The method expects a string as input data. If non-string data is received,
        it will log a warning and return an 'unclassified' intent.
        The `context` dictionary can optionally contain an 'intent_mapping' key
        to override the default keyword-intent mapping.

        Args:
            data (Any): The input data, typically a string representing user input.
            context (Dict[str, Any]): A dictionary for shared context or configuration.
                                       May include a 'intent_mapping' key
                                       (Dict[str, List[str]]) to customize intent rules.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'original_input': The raw input data.
                            - 'classified_intent': The identified intent (e.g., "book_flight",
                                                   "unclassified").
                            - 'confidence': A simulated confidence score (1.0 for a match,
                                            0.0 for unclassified).
        """
        if not isinstance(data, str):
            logger.warning(
                "Received non-string data for intent classification. Type: %s. Returning 'unclassified'.",
                type(data).__name__
            )
            return {
                "original_input": data,
                "classified_intent": "unclassified",
                "confidence": 0.0
            }

        text_input = data.lower()
        classified_intent = "unclassified"
        confidence = 0.0

        # Prioritize custom mapping provided in context
        intent_mapping = context.get("intent_mapping", self._DEFAULT_INTENT_MAPPING)

        # Simple keyword-based classification logic
        # This will return the first intent whose keywords are found.
        # For a more robust solution, scoring and disambiguation would be necessary.
        for intent, keywords in intent_mapping.items():
            for keyword in keywords:
                if keyword.lower() in text_input:
                    classified_intent = intent
                    confidence = 1.0  # Simple simulation: 1.0 if any match is found
                    logger.debug("Input '%s' matched keyword '%s' for intent '%s'.", data, keyword, intent)
                    return {
                        "original_input": data,
                        "classified_intent": classified_intent,
                        "confidence": confidence
                    }
        
        logger.info("No specific intent found for input: '%s'. Defaulting to 'unclassified'.", data)
        return {
            "original_input": data,
            "classified_intent": classified_intent,
            "confidence": confidence
        }