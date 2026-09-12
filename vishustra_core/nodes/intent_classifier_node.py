import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent
    of a given text input (e.g., a user query).

    This node uses a simplified keyword-based approach for intent detection
    and can be configured via the context dictionary for custom classification rules
    and fallback intent behavior.
    """

    def __init__(self) -> None:
        """
        Initializes the IntentClassifierNode with a set of default
        classification rules. These can be overridden or augmented
        via the `context` dictionary during the `process` call.
        """
        self._default_classification_rules = {
            "greet": ["hello", "hi", "hey", "good morning", "good evening"],
            "track_order": ["track order", "where is my", "order status", "shipment status"],
            "cancel_order": ["cancel order", "undo purchase", "stop order"],
            "make_purchase": ["buy", "purchase", "order", "get me"],
            "support": ["help", "support", "customer service", "agent"]
        }
        self._default_fallback_intent = "unknown"
        logger.debug(f"{self.node_name} initialized with default classification rules.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The `data` is expected to be a string representing a user query or
        a similar piece of text. The node iterates through predefined (or
        context-provided) rules to find a matching intent.

        Args:
            data: The input data to be processed, expected to be a string.
            context: A dictionary providing runtime configuration.
                     Recognized keys:
                     - 'classification_rules' (Dict[str, List[str]]): Overrides
                       or extends the default keyword-based classification rules.
                       Format: `{'intent_name': ['keyword1', 'keyword2']}`.
                     - 'fallback_intent' (str): Specifies the intent to return
                       if no specific rule matches. Defaults to "unknown".

        Returns:
            A dictionary containing the classified 'intent' (str) and a
            'confidence' (float) score.

        Raises:
            TypeError: If the input `data` is not a string.
            Exception: For any unforeseen errors during the classification process.
        """
        if not isinstance(data, str):
            logger.error(
                f"{self.node_name}: Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'.")
            raise TypeError(
                f"IntentClassifierNode expects string data, "
                f"but received {type(data).__name__}.")

        query_text = data.lower().strip()
        classified_intent = self._default_fallback_intent
        confidence = 0.0

        try:
            # Retrieve classification rules from context, falling back to defaults
            classification_rules = context.get(
                'classification_rules', self._default_classification_rules)
            
            # Retrieve fallback intent from context
            fallback_intent = context.get(
                'fallback_intent', self._default_fallback_intent)

            # Attempt to match intent based on keywords
            for intent_name, keywords in classification_rules.items():
                for keyword in keywords:
                    if keyword in query_text:
                        classified_intent = intent_name
                        confidence = 0.95  # High confidence for a direct keyword match
                        logger.debug(
                            f"{self.node_name}: Matched keyword '{keyword}' for intent "
                            f"'{intent_name}' in query '{query_text}'.")
                        break  # Found a match, no need to check other keywords for this intent
                if classified_intent != fallback_intent:
                    break  # Found a match for an intent, no need to check other intents

            # Assign lower confidence if fallback was used
            if classified_intent == fallback_intent:
                confidence = 0.5
                logger.info(
                    f"{self.node_name}: No specific intent matched for query '{query_text}'. "
                    f"Falling back to '{fallback_intent}'.")
            else:
                logger.info(
                    f"{self.node_name}: Classified intent '{classified_intent}' for query "
                    f"'{query_text}' with confidence {confidence:.2f}.")

            return {"intent": classified_intent, "confidence": confidence}

        except Exception as e:
            logger.exception(
                f"{self.node_name}: An unexpected error occurred during intent "
                f"classification for query '{query_text}'.")
            raise Exception(f"Failed to classify intent: {e}") from e