import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of an input text.

    This node simulates intent classification, typically using a keyword-based
    approach or by integrating with an external intent recognition service/model.
    In a production environment, this would involve loading and running a
    machine learning model (e.g., a fine-tuned transformer model or a simpler
    rules engine).

    Configuration can be provided during initialization or dynamically via
    the `context` dictionary during processing.
    """

    def __init__(self, classification_rules: Dict[str, str] = None, default_intent: str = "unrecognized"):
        """
        Initializes the IntentClassifierNode with optional classification rules
        and a default intent.

        Args:
            classification_rules (Dict[str, str], optional): A dictionary
                where keys are keywords (or phrases) and values are the
                corresponding intent names (e.g., {"hello": "greeting"}).
                Keys are matched in a case-insensitive manner against the input text.
                If None, a basic set of internal rules is used.
            default_intent (str, optional): The intent to return if no
                specific intent is matched by the provided rules.
                Defaults to "unrecognized".
        """
        # Internal default rules to provide a baseline if none are specified
        self._internal_default_rules = {
            "hello": "greeting", "hi": "greeting", "hey": "greeting",
            "buy": "purchase_intent", "order": "purchase_intent", "purchase": "purchase_intent",
            "support": "support_request", "help": "support_request", "issue": "support_request",
            "cancel": "cancellation_request", "undo": "cancellation_request",
            "thanks": "acknowledgement", "thank you": "acknowledgement",
            "faq": "information_query", "question": "information_query", "how to": "information_query",
            "price": "pricing_query", "cost": "pricing_query",
        }

        # Combine provided rules with internal defaults, allowing user rules to override
        self._classification_rules = self._internal_default_rules.copy()
        if classification_rules:
            self._classification_rules.update({k.lower(): v for k, v in classification_rules.items()})

        self._default_intent = default_intent
        logger.debug(f"IntentClassifierNode initialized with rules: {list(self._classification_rules.keys())} and default intent: '{self._default_intent}'")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text data.

        The node expects the `data` to be a string representing the user's
        utterance or a piece of text to be analyzed. It uses a keyword-matching
        logic, enhanced by rules potentially provided in the `context`.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
                        If not a string, an 'error_invalid_input_type' intent is returned.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                     This can be used to dynamically pass additional
                                     `intent_classification_rules` or other configuration
                                     for a more complex classification model.

        Returns:
            Dict[str, Any]: A dictionary containing the 'intent' (str) and a
                            simulated 'confidence' (float) for the classification.
                            Example: `{"intent": "greeting", "confidence": 0.95}`.
                            For invalid input types, it returns:
                            `{"intent": "error_invalid_input_type", "confidence": 0.0}`.
                            For an empty text input or no match, it returns the
                            `default_intent` with lower confidence.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received non-string data of type: {type(data).__name__}. Expected 'str'.")
            return {"intent": "error_invalid_input_type", "confidence": 0.0}

        text = data.lower().strip()
        logger.debug(f"Attempting to classify intent for text: '{text}'")

        # Create a mutable copy of rules, allowing context to provide dynamic overrides/additions
        current_rules = self._classification_rules.copy()
        if 'intent_classification_rules' in context and isinstance(context['intent_classification_rules'], dict):
            context_rules = {k.lower(): v for k, v in context['intent_classification_rules'].items()}
            current_rules.update(context_rules)
            logger.debug(f"Dynamic intent rules from context applied. Total rules: {len(current_rules)}.")

        matched_intent = self._default_intent
        confidence = 0.1  # Default low confidence for unrecognized or empty intent

        if not text:
            logger.debug("Input text is empty. Returning default intent.")
            return {"intent": self._default_intent, "confidence": confidence}

        # Perform keyword-based intent classification
        for keyword, intent in current_rules.items():
            if keyword in text:
                matched_intent = intent
                confidence = 0.95  # Simulate high confidence for a direct keyword match
                logger.info(f"Intent classified as '{matched_intent}' based on keyword '{keyword}'.")
                break # Return the first matching intent found

        if matched_intent == self._default_intent:
            logger.info(f"No specific intent matched for text: '{text}'. Returning default intent: '{self._default_intent}'.")

        return {"intent": matched_intent, "confidence": confidence}
