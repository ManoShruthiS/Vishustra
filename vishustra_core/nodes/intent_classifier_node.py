import logging
from typing import Any, Dict, List, Literal

# Ensure the base_node path is correct according to Vishustra's structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# Define common intent types that this classifier might produce
IntentType = Literal[
    "booking_reservation",
    "cancellation",
    "customer_support",
    "product_inquiry",
    "account_management",
    "greeting",
    "unknown_intent"
]

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the user's intent
    from an input text utterance.

    This node simulates intent classification based on predefined keywords.
    In a production environment, this would typically integrate with an
    external Natural Language Understanding (NLU) service or a sophisticated
    machine learning model.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode.
        Sets up predefined keyword mappings for demonstration purposes.
        These mappings provide a rudimentary way to detect user intents.
        """
        # In a real-world scenario, these mappings would likely be more complex,
        # loaded from configuration files, a database, or dynamically derived
        # from a trained NLU model.
        self._intent_keywords: Dict[IntentType, List[str]] = {
            "booking_reservation": ["book", "reserve", "schedule", "appointment", "make a booking"],
            "cancellation": ["cancel", "undo", "reschedule", "revoke", "change my booking"],
            "customer_support": ["help", "support", "issue", "problem", "contact", "agent"],
            "product_inquiry": ["what is", "tell me about", "features", "price", "info on", "details about"],
            "account_management": ["my account", "login", "password", "profile", "settings", "username"],
            "greeting": ["hello", "hi", "hey", "good morning", "good evening", "greetings"]
        }
        logger.info("IntentClassifierNode initialized with predefined intent mappings.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> IntentType:
        """
        Processes the input data, expected to be a user utterance string,
        and classifies its intent based on predefined keywords.

        Args:
            data: The input data, expected to be a string representing a user utterance.
            context: A dictionary containing contextual information.
                     Currently not directly used for classification logic within
                     this simulation but available for future extensions (e.g.,
                     passing model parameters, session state).

        Returns:
            A string representing the classified intent (e.g., "booking_reservation",
            "customer_support", "unknown_intent").

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string after stripping whitespace.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for IntentClassifierNode. "
                f"Expected str, got {type(data).__name__}."
            )
            raise TypeError(
                f"IntentClassifierNode expects string input, "
                f"but received {type(data).__name__}"
            )

        utterance = data.strip().lower()
        if not utterance:
            logger.warning(
                "Received an empty or whitespace-only utterance "
                "for intent classification."
            )
            raise ValueError("IntentClassifierNode received an empty utterance.")

        logger.debug(f"Starting intent classification for utterance: '{utterance}'")

        detected_intent: IntentType = "unknown_intent"
        
        # Perform a simple keyword-based classification
        for intent, keywords in self._intent_keywords.items():
            for keyword in keywords:
                if keyword in utterance:
                    detected_intent = intent
                    logger.debug(
                        f"Found keyword '{keyword}' for intent '{intent}' "
                        f"in utterance: '{utterance}'."
                    )
                    break  # Found a keyword for this intent, no need to check others for this intent
            if detected_intent != "unknown_intent":
                break  # An intent has been found, no need to check other intents

        if detected_intent == "unknown_intent":
            logger.info(f"No specific intent detected for utterance: '{utterance}'. "
                        "Classified as 'unknown_intent'.")
        else:
            logger.info(f"Utterance: '{utterance}' classified as intent: '{detected_intent}'.")
            
        return detected_intent