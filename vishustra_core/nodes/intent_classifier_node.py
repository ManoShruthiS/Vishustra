import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node designed to classify the underlying intent of a given text input.

    This node simulates intent classification based on a predefined set of keywords.
    In a production environment, this would typically integrate with a dedicated
    Natural Language Understanding (NLU) service or a fine-tuned machine learning model.
    """

    _INTENT_MAP = {
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
        "product_inquiry": ["what products", "product list", "show me products", "details about", "catalogue"],
        "order_status": ["order status", "where is my order", "track my order", "delivery time"],
        "support_request": ["help", "support", "technical issue", "contact support", "assistance"],
        "goodbye": ["bye", "goodbye", "see you", "farewell"],
        "acknowledgment": ["thank you", "thanks", "appreciate it"],
    }
    _DEFAULT_INTENT = "general_query"

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """
        Classifies the intent of the input text data.

        The `data` input is expected to be a string representing a user query or utterance.
        The `context` dictionary provides a mechanism for passing operational metadata
        or configuration relevant to the node's execution, though it is not
        directly utilized for the classification logic in this simulated implementation.

        Args:
            data: The input text to be classified. Must be a string.
            context: A dictionary containing contextual information for the node's operation.
                     (e.g., session ID, user preferences, global configurations).

        Returns:
            A dictionary containing the original query, the classified intent, and a
            simulated confidence score. Example:
            `{"original_query": "Hello there", "classified_intent": "greeting", "confidence": 0.95}`

        Raises:
            TypeError: If the input `data` is not of type string.
            ValueError: If the input `data` is an empty string after stripping whitespace.
        """
        logger.debug(f"[{self.node_name}] Attempting intent classification for data: '{data}'")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' for intent classification must be a string. "
                f"Received type '{type(data).__name__}'."
            )

        query = data.strip().lower()

        if not query:
            logger.warning(
                f"[{self.node_name}] Received an empty query string after stripping whitespace. "
                "Unable to perform intent classification."
            )
            raise ValueError(
                f"[{self.node_name}] Input query string cannot be empty for intent classification."
            )

        classified_intent = self._DEFAULT_INTENT
        confidence = 0.5  # Base confidence for unmatched or general queries

        # Simple keyword-based classification
        for intent, keywords in self._INTENT_MAP.items():
            for keyword in keywords:
                if keyword in query:
                    classified_intent = intent
                    confidence = 0.95  # Higher confidence for a positive match
                    logger.debug(
                        f"[{self.node_name}] Detected keyword '{keyword}' indicating intent '{intent}'."
                    )
                    # For this simulation, we take the first matching keyword and intent
                    # In a real system, this would involve more sophisticated scoring.
                    break
            if classified_intent != self._DEFAULT_INTENT:
                break # If an intent was found, no need to check further intent categories

        logger.info(
            f"[{self.node_name}] Classified intent for query '{data[:75]}{'...' if len(data) > 75 else ''}' "
            f"as '{classified_intent}' with a confidence of {confidence:.2f}."
        )

        return {
            "original_query": data,
            "classified_intent": classified_intent,
            "confidence": confidence,
        }