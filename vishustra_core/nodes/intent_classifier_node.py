
import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent of a given
    text input. This implementation uses a keyword-based matching strategy for
    demonstration and initial prototyping.
    """

    def __init__(self, intent_map: Dict[str, List[str]] = None):
        """
        Initializes the IntentClassifierNode with a predefined set of intent rules.

        Args:
            intent_map (Dict[str, List[str]], optional): A dictionary where keys are intent names
                                                         and values are lists of keywords associated
                                                         with that intent. If None, a default map is used.
                                                         Keywords are case-insensitive.
        """
        self._intent_map = intent_map if intent_map is not None else {
            "greet": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "farewell": ["bye", "goodbye", "see you", "later", "cya"],
            "ask_balance": ["balance", "account", "funds", "how much money", "what's in my account"],
            "transfer_money": ["transfer", "send money", "move funds", "pay someone"],
            "order_status": ["order status", "my order", "where is my package", "tracking"],
            "customer_service": ["help", "support", "speak to an agent", "contact us"],
        }
        logger.debug(f"IntentClassifierNode initialized with intent map: {self._intent_map}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text using predefined keyword rules.

        Args:
            data (Any): The input data, expected to be a string utterance from the user.
            context (Dict[str, Any]): The processing context, which may contain session-specific
                                      information or previous node outputs. Not directly
                                      utilized by this node for classification logic, but
                                      passed along in the framework.

        Returns:
            Dict[str, Any]: A dictionary containing the classification results, including:
                            - "original_utterance": The unprocessed input string.
                            - "classified_intent": The determined intent (e.g., "greet", "fallback").
                            - "confidence": A numerical confidence score for the classification.
                            - "processed_by": The name of this node for traceability.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping whitespace.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Received non-string data: {type(data)}. Expected 'str'."
            )
            raise TypeError(
                f"{self.node_name} expects string input for 'data', but received {type(data)}."
            )

        utterance = data.strip().lower()

        if not utterance:
            logger.warning(
                f"[{self.node_name}] Received an empty or whitespace-only utterance."
            )
            raise ValueError(f"{self.node_name} cannot process an empty utterance.")

        classified_intent = "fallback"
        confidence = 0.50  # Default low confidence for fallback scenarios
        
        # Iterate through defined intents and their keywords to find a match
        for intent, keywords in self._intent_map.items():
            for keyword in keywords:
                if keyword in utterance:
                    classified_intent = intent
                    confidence = 0.95  # High confidence for a direct keyword match
                    logger.info(
                        f"[{self.node_name}] Classified intent '{classified_intent}' "
                        f"for utterance '{data}' based on keyword '{keyword}'."
                    )
                    # Once an intent is found, we stop searching.
                    # For more complex scenarios, one might implement scoring.
                    break 
            if classified_intent != "fallback":
                break # Exit outer loop if an intent was found

        if classified_intent == "fallback":
            logger.info(
                f"[{self.node_name}] No specific intent matched for utterance '{data}'. "
                f"Falling back to 'fallback' intent."
            )

        # Append classification results to the context or return a new structure
        return {
            "original_utterance": data,
            "classified_intent": classified_intent,
            "confidence": confidence,
            "processed_by": self.node_name,
        }

