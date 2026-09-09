import logging
from typing import Any, Dict

# Assuming this import path based on the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent
    of an incoming text input, typically a user query or utterance.

    This node simulates intent classification based on keyword matching
    against a configurable set of rules. It takes a string as input and
    produces a dictionary containing the classified intent and a
    simulated confidence score.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its underlying intent.

        The node expects `data` to be a string representing the query
        or utterance. It uses a predefined or context-provided set of
        keyword-to-intent mappings for classification.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual
                information. Can include 'intent_rules' (Dict[str, List[str]])
                to override default classification rules.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'intent' (str): The classified intent (e.g., "greeting", "place_order").
                - 'confidence' (float): A simulated confidence score (0.0 to 1.0).
                - 'message' (str, optional): An error message if processing fails.

        Raises:
            TypeError: If the input `data` is not a string, indicating an
                       invalid input type for this node.
        """
        if not isinstance(data, str):
            error_msg = (
                f"IntentClassifierNode expected string input, but received type "
                f"'{type(data).__name__}'. Value: '{data}'"
            )
            logger.error(error_msg)
            # Returning a structured error allows downstream nodes to handle it gracefully
            return {"intent": "error", "confidence": 0.0, "message": error_msg}

        text_input = data.lower().strip()
        logger.debug("IntentClassifierNode received text for classification: '%s'", text_input)

        # Define default intent classification rules (keywords mapping to intents)
        # These can be overridden or extended via the 'context' dictionary.
        default_intent_rules = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
            "farewell": ["bye", "goodbye", "see you", "later", "thanks bye"],
            "ask_about_product": ["product", "item", "what do you sell", "catalog", "show products"],
            "place_order": ["order", "buy", "purchase", "checkout", "add to cart"],
            "check_status": ["status", "track", "where is my", "order progress"],
            "customer_support": ["help", "support", "contact us", "problem", "speak to agent"],
            "thank_you": ["thank you", "thanks", "appreciate it"],
        }

        # Merge default rules with any rules provided in the context,
        # allowing context rules to take precedence or add new ones.
        intent_rules = {**default_intent_rules, **context.get("intent_rules", {})}

        classified_intent = "unknown"
        confidence = 0.0

        for intent, keywords in intent_rules.items():
            for keyword in keywords:
                if keyword in text_input:
                    classified_intent = intent
                    confidence = 1.0  # Simulate high confidence for a direct keyword match
                    logger.info("Intent classified as '%s' based on keyword '%s'", intent, keyword)
                    break  # Found a match for this intent, move to the next
            if classified_intent != "unknown":
                break  # Found a specific intent, no need to check further rules

        if classified_intent == "unknown":
            logger.info("No specific intent found for text: '%s'. Defaulting to 'unknown'.", text_input)
            # Assign a lower confidence for unclassified inputs
            confidence = 0.1

        return {"intent": classified_intent, "confidence": confidence}
