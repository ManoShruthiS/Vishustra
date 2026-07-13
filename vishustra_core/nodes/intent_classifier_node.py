import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node designed to classify the intent of a given text input.
    This node simulates intent classification based on predefined keyword patterns.
    In a production environment, this would typically integrate with a dedicated
    intent recognition model or an LLM service.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode.
        Sets up a mapping of keywords to simulate different user intents.
        """
        self._intent_patterns: Dict[str, list[str]] = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "track_order": ["track my order", "where is my order", "order status", "shipment update"],
            "place_order": ["i want to order", "buy product", "make a purchase", "new order"],
            "cancel_order": ["cancel my order", "return item", "undo purchase"],
            "product_info": ["tell me about", "what is", "product details", "specifications"],
            "customer_support": ["speak to an agent", "need help", "support representative", "contact support"],
            "thank_you": ["thanks", "thank you very much", "appreciate it"],
            "goodbye": ["bye", "goodbye", "see you later"],
        }
        logger.info("IntentClassifierNode initialized with keyword-based intent patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (user utterance) to classify its intent.

        Args:
            data: The input data, expected to be a string representing the user's utterance.
            context: A dictionary containing shared context information that can be
                     read from or written to by nodes in the orchestration pipeline.

        Returns:
            The classified intent as a string.

        Raises:
            ValueError: If the input data is not a string or is empty.
        """
        if not isinstance(data, str):
            error_msg = (f"[{self.node_name}] Invalid input type. Expected 'str', "
                         f"received '{type(data).__name__}'.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not data.strip():
            error_msg = f"[{self.node_name}] Input data cannot be an empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)

        input_text = data.lower()
        classified_intent = "unknown_intent"
        confidence_score = 0.0  # Simulated confidence

        logger.debug(f"[{self.node_name}] Attempting to classify intent for input: '{data}'")

        # Iterate through defined intent patterns to find a match
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern in input_text:
                    classified_intent = intent
                    confidence_score = 0.85  # Assign a fixed confidence for simplicity in simulation
                    logger.info(
                        f"[{self.node_name}] Classified intent as '{classified_intent}' "
                        f"based on pattern: '{pattern}'."
                    )
                    break  # Found a match for this intent
            if classified_intent != "unknown_intent":
                break  # Found an intent, no need to check further

        if classified_intent == "unknown_intent":
            logger.warning(
                f"[{self.node_name}] No specific intent classified for input: '{data}'. "
                f"Defaulting to '{classified_intent}'."
            )
            # A lower confidence for unknown intents
            confidence_score = 0.5 if not input_text else 0.2

        # Update the context dictionary with the classification result
        context["classified_intent"] = classified_intent
        context["intent_confidence"] = confidence_score

        logger.debug(
            f"[{self.node_name}] Context updated: 'classified_intent'='{classified_intent}', "
            f"'intent_confidence'={confidence_score:.2f}."
        )

        return classified_intent
