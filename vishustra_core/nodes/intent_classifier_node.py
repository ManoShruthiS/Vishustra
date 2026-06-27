import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a user query.

    This node simulates intent classification based on predefined keyword rules.
    In a production environment, this would typically leverage an external LLM
    or a dedicated machine learning model for robust intent recognition.
    """

    def __init__(self, classification_rules: Dict[str, List[str]] = None):
        """
        Initializes the IntentClassifierNode with optional custom classification rules.

        Args:
            classification_rules: An optional dictionary where keys are intent names (str)
                                  and values are lists of keywords (str) associated
                                  with that intent. If None, a set of default rules is used.
        """
        super().__init__()
        self._default_rules: Dict[str, List[str]] = {
            "book_flight": ["book flight", "flight ticket", "travel to", "reservation"],
            "check_status": ["flight status", "where is my flight", "delay", "arrival time"],
            "cancel_order": ["cancel order", "return item", "refund"],
            "general_query": ["hello", "hi", "how are you", "what can you do"],
            "account_info": ["my account", "login", "password", "profile"]
        }
        self.classification_rules = classification_rules if classification_rules is not None else self._default_rules
        logger.info(f"Initialized {self.node_name} with rules for intents: {list(self.classification_rules.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a user query string) to classify its intent.

        This method performs a simulated intent classification by checking for keywords
        in the input data against the configured classification rules.

        Args:
            data: The input data, expected to be a string representing a user query.
            context: A dictionary containing contextual information for the processing.
                     While not directly used for keyword matching in this simulation,
                     it would typically carry parameters like session ID, user history,
                     or LLM configuration (e.g., model name, temperature) if an LLM
                     were used for classification.

        Returns:
            A dictionary containing the classification results:
            - 'original_query': The input query string.
            - 'classified_intent': The identified intent as a string (e.g., "book_flight", "unknown").
            - 'confidence': A simulated confidence score (e.g., 1.0 if a match, 0.0 otherwise).
            - 'explanation': A brief description of how the intent was derived.

            If the input `data` is not a string, an error dictionary is returned.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {data!r}"
            )
            return {
                "error": "InvalidInputTypeError",
                "message": "Input data for intent classification must be a string.",
                "received_type": str(type(data)),
                "original_data": data
            }

        query = data.lower().strip()
        if not query:
            logger.warning(f"[{self.node_name}] Received an empty or whitespace-only query string.")
            return {
                "original_query": data,
                "classified_intent": "empty_query",
                "confidence": 0.0,
                "explanation": "Input query was empty or contained only whitespace."
            }

        identified_intent: str = "unknown"
        confidence: float = 0.0
        explanation: str = "No specific intent found."

        # Simulate intent classification using keyword matching
        for intent_name, keywords in self.classification_rules.items():
            for keyword in keywords:
                if keyword in query:
                    identified_intent = intent_name
                    confidence = 1.0  # Simple simulation: 1.0 if any keyword matches
                    explanation = f"Matched keyword: '{keyword}' for intent '{intent_name}'."
                    logger.debug(
                        f"[{self.node_name}] Classified query '{query}' as '{identified_intent}' "
                        f"due to keyword '{keyword}'."
                    )
                    break  # Found a match, no need to check other keywords for this intent
            if identified_intent != "unknown":
                break  # Found a match, no need to check other intents

        if identified_intent == "unknown":
            logger.info(f"[{self.node_name}] Could not classify intent for query: '{query}'. Defaulting to 'unknown'.")
            explanation = "No matching keywords found for any defined intent in the query."

        result: Dict[str, Any] = {
            "original_query": data,
            "classified_intent": identified_intent,
            "confidence": confidence,
            "explanation": explanation
        }
        return result
