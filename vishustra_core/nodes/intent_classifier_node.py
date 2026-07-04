import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.
    This node uses a keyword-based approach for demonstration, but in a
    production environment, it would typically integrate with advanced
    Natural Language Understanding (NLU) models.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the name of the node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text based on predefined rules or
        rules provided in the context.

        Args:
            data (Any): The input data, expected to be a string containing
                        the user's query or utterance.
            context (Dict[str, Any]): A dictionary containing contextual
                                      information. Can include 'intent_rules'
                                      to dynamically define or override
                                      classification rules.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent
                            and a confidence score.
                            Example: {"intent": "greeting", "confidence": 0.95}

        Raises:
            ValueError: If the input data is not a string, as this node
                        specifically operates on textual input.
        """
        if not isinstance(data, str):
            logger.error(
                f"IntentClassifierNode received non-string data: type={type(data)}. "
                "Expected a string for intent classification."
            )
            raise ValueError("IntentClassifierNode expects string input for classification.")

        text_input = data.lower().strip()
        classified_intent = "unknown"
        confidence = 0.5  # Default confidence for 'unknown' or general matches

        # Base keyword-to-intent mappings
        # In a real system, these would be loaded from configuration or a model.
        base_intent_rules = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
            "order_status": ["where is my order", "track my order", "order status", "my order number", "delivery status"],
            "product_info": ["tell me about", "what is", "product details", "features of", "specification for"],
            "farewell": ["bye", "goodbye", "see you", "later"],
            "thank_you": ["thank you", "thanks", "appreciate it", "cheers"],
            "help": ["help me", "i need help", "support", "assist me"],
            "cancel_order": ["cancel my order", "stop order", "revoke order"]
        }

        # Merge base rules with any context-provided rules.
        # Context rules take precedence or extend the base rules.
        context_intent_rules = context.get("intent_rules", {})
        merged_intent_rules = {**base_intent_rules, **context_intent_rules}

        # Perform a simple keyword-based classification
        for intent, keywords in merged_intent_rules.items():
            for keyword in keywords:
                if keyword in text_input:
                    classified_intent = intent
                    confidence = 0.9  # Higher confidence for a direct keyword match
                    logger.debug(
                        f"Intent '{classified_intent}' identified by keyword '{keyword}' "
                        f"in input: '{text_input}'."
                    )
                    break # Matched an intent, no need to check more keywords for this intent
            if classified_intent != "unknown":
                break # Matched an intent, no need to check other intents

        if classified_intent == "unknown":
            confidence = 0.3  # Lower confidence for unclassified inputs
            logger.info(
                f"No specific intent identified for input: '{text_input}'. "
                f"Classified as '{classified_intent}'."
            )
        else:
            logger.info(
                f"Input '{text_input}' classified as intent '{classified_intent}' "
                f"with confidence {confidence:.2f}."
            )

        return {"intent": classified_intent, "confidence": confidence}
